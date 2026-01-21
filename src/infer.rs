use anyhow::Result;
use rand::Rng;

use burn::module::Module;
use burn::record::{DefaultRecorder, FileRecorder};
use burn::tensor::{backend::Backend, Data, ElementConversion, Shape, Tensor, Int};

use crate::checkpoint::load_config;
use crate::data::tokenizer::ByteTokenizer;
use crate::model::gpt::Gpt;
use crate::settings::{AppSettings, ChatSettings};

use crate::memory::{
    embed::HashEmbedder,
    retrieve::{top_k_chunks, top_k_notes},
    store::MemoryStore,
    prompt::build_working_memory_prompt,
    consolidate::{consolidate_semantic_notes, EvidenceChunk},
};

type BackendImpl = burn_ndarray::NdArray<f32>;

// Compute a softmax distribution with optional temperature scaling.
fn softmax_vec(logits: &[f32], temperature: f64) -> Vec<f32> {
    if temperature <= 0.0 {
        let mut out = vec![0.0; logits.len()];
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] { best = i; }
        }
        out[best] = 1.0;
        return out;
    }
    let temp = temperature as f32;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| ((x - max) / temp).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-8);
    exps.iter().map(|x| x / sum).collect()
}

// Sample an index from the top-k probabilities.
fn sample_top_k(probs: &[f32], k: usize, rng: &mut impl Rng) -> usize {
    let mut idxs: Vec<usize> = (0..probs.len()).collect();
    idxs.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let k = k.min(idxs.len());
    let top = &idxs[..k];

    let mut mass = 0.0f32;
    for &i in top { mass += probs[i]; }
    let mut r = rng.gen::<f32>() * mass;
    for &i in top {
        r -= probs[i];
        if r <= 0.0 { return i; }
    }
    top[k-1]
}

// Autoregressively generate tokens from a prompt.
// Autoregressively generate tokens from a prompt.
fn generate(model: &Gpt<BackendImpl>, cfg: &crate::config::ModelConfig, prompt: &str, max_new_tokens: usize, temperature: f64, top_k: usize) -> String {
    let device = burn_ndarray::NdArrayDevice::default();
    let mut ids = ByteTokenizer::encode(prompt);
    let mut rng = rand::thread_rng();

    for _ in 0..max_new_tokens {
        let start = ids.len().saturating_sub(cfg.seq_len);
        let window = &ids[start..];

        let mut input = vec![0u32; cfg.seq_len];
        let offset = cfg.seq_len - window.len();
        input[offset..].copy_from_slice(window);

        let x = Tensor::<BackendImpl, 2, Int>::from_data(
            Data::new(
                input
                    .iter()
                    .map(|&v| v.elem::<<BackendImpl as Backend>::IntElem>())
                    .collect::<Vec<_>>(),
                Shape::new([1, cfg.seq_len]),
            ),
            &device,
        );

        let logits = model.forward(x); // [1, seq, vocab]
        let logits_last = logits.slice([0..1, (cfg.seq_len-1)..cfg.seq_len, 0..cfg.vocab]).reshape([cfg.vocab]);

        let logits_vec = logits_last.into_data().value;
        let probs = softmax_vec(&logits_vec, temperature);
        let next = if top_k == 0 {
            probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
        } else {
            sample_top_k(&probs, top_k, &mut rng)
        };

        ids.push(next as u32);
    }

    ByteTokenizer::decode(&ids)
}

/// Chat with optional long-term memory.
/// Short-term memory = model context window (cfg.seq_len bytes).
/// Long-term memory = episodic chunk store + semantic notes store.
/// We retrieve relevant memories and inject them into the prompt (working memory).
// Chat loop that optionally augments the prompt with long-term memory.
// Chat loop that optionally augments the prompt with long-term memory.
pub fn run_chat_with_memory(
    ckpt_dir: &str,
    memory_dir: Option<&str>,
    user_prompt: Option<&str>,
    repl: bool,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    k_episodic: usize,
    k_semantic: usize,
    wm_words: usize,
    consolidate: bool,
    settings_path: Option<&str>,
) -> Result<()> {
    let cfg = load_config(ckpt_dir)?;
    let device = burn_ndarray::NdArrayDevice::default();
    let settings = match settings_path {
        Some(path) => Some(AppSettings::load(path)?),
        None => None,
    };
    let chat_settings = settings.as_ref().and_then(|s| s.chat.clone());

    // NOTE: model weight re-loading is still TODO (see README).
    // This creates a model with the saved config so the program builds and runs.
    let mut model: Gpt<BackendImpl> = Gpt::new(&cfg, &device);
    if let Some(path) = latest_checkpoint_path(ckpt_dir) {
        let recorder = DefaultRecorder::new();
        if let Ok(loaded) = model.clone().load_file(path, &recorder, &device) {
            model = loaded;
        }
    }

    let mut final_prompt = user_prompt.unwrap_or("").to_string();
    let mut store_opt: Option<MemoryStore> = None;
    let mut embedder_opt: Option<HashEmbedder> = None;
    let mut sources: Vec<String> = Vec::new();

    if let Some(memdir) = memory_dir {
        let store = MemoryStore::load(memdir)?;
        // read embedder config (hash dim) if present, else default
        let dim = 2048usize;
        let embedder = HashEmbedder::new(dim);
        let q_emb = embedder.embed(&final_prompt);

        let epi = top_k_chunks(&q_emb, &store.chunks, k_episodic);
        let sem = top_k_notes(&q_emb, &store.notes, k_semantic);

        let episodic_lines: Vec<String> = epi.iter().map(|(c, _s)| {
            sources.push(c.id.clone());
            format!("[{}:{}] {}", c.book, c.offset, trim_text(&c.text, 60))
        }).collect();

        let semantic_lines: Vec<String> = sem.iter().map(|(n, _s)| n.text.clone()).collect();

        final_prompt = build_working_memory_prompt(&final_prompt, &semantic_lines, &episodic_lines, wm_words);

        store_opt = Some(store);
        embedder_opt = Some(embedder);
    }

    let (temperature, top_k, max_new_tokens, system_prompt) =
        apply_chat_settings(temperature, top_k, max_new_tokens, chat_settings);

    if repl {
        run_repl(
            &model,
            &cfg,
            memory_dir,
            store_opt,
            embedder_opt,
            k_episodic,
            k_semantic,
            wm_words,
            consolidate,
            system_prompt,
            temperature,
            top_k,
            max_new_tokens,
        )?;
        return Ok(());
    }

    if final_prompt.trim().is_empty() {
        anyhow::bail!("--prompt is required unless --repl is set");
    }

    let prompt = prepend_system_prompt(&final_prompt, system_prompt.as_deref());
    let out = generate(&model, &cfg, &prompt, max_new_tokens, temperature, top_k);
    println!("{out}");

    // Very simple consolidation: store Q/A as a semantic note.
    if consolidate {
        if let (Some(mut store), Some(embedder), Some(memdir)) = (store_opt, embedder_opt, memory_dir) {
            let evidence_chunks: Vec<EvidenceChunk> = sources.iter().filter_map(|id| {
                store.chunks.iter().find(|c| c.id == *id).map(|c| EvidenceChunk {
                    id: c.id.clone(),
                    text: c.text.clone(),
                })
            }).collect();

            let new_notes = consolidate_semantic_notes(
                &embedder,
                user_prompt.unwrap_or(""),
                &out,
                &evidence_chunks,
                &store.notes,
                store.notes.len(),
            );

            if !new_notes.is_empty() {
                store.notes.extend(new_notes);
                store.save_notes(memdir)?;
            }
        }
    }

    Ok(())
}

// Trim a string to a fixed number of words.
fn trim_text(text: &str, max_words: usize) -> String {
    text.split_whitespace().take(max_words).collect::<Vec<_>>().join(" ")
}
// Find the most recent model checkpoint file in the checkpoint directory.
fn latest_checkpoint_path(ckpt_dir: &str) -> Option<std::path::PathBuf> {
    let dir = std::path::Path::new(ckpt_dir).join("checkpoint");
    let _recorder = DefaultRecorder::new();
    let ext = <DefaultRecorder as FileRecorder<BackendImpl>>::file_extension();
    let mut best_epoch = None;
    let mut best_path = None;

    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy();
        if !name.starts_with("model-") || !name.ends_with(ext) {
            continue;
        }
        let epoch_str = name.trim_start_matches("model-").trim_end_matches(&format!(".{ext}"));
        if let Ok(epoch) = epoch_str.parse::<usize>() {
            if best_epoch.map_or(true, |best| epoch > best) {
                best_epoch = Some(epoch);
                best_path = Some(path);
            }
        }
    }
    best_path
}

// Apply optional chat settings and emotional presets to generation params.
fn apply_chat_settings(
    temperature: f64,
    top_k: usize,
    max_new_tokens: usize,
    settings: Option<ChatSettings>,
) -> (f64, usize, usize, Option<String>) {
    let mut temp = temperature;
    let mut k = top_k;
    let mut max_new = max_new_tokens;
    let mut system_prompt = None;

    if let Some(cfg) = settings {
        if let Some(preset) = cfg.emotional_preset.as_deref() {
            if preset.eq_ignore_ascii_case("warm") || preset.eq_ignore_ascii_case("empathetic") {
                system_prompt = Some("You are a warm, empathetic assistant. Keep responses supportive and human.".to_string());
                temp = 0.9;
                k = 60;
            }
        }
        if let Some(v) = cfg.temperature { temp = v; }
        if let Some(v) = cfg.top_k { k = v; }
        if let Some(v) = cfg.max_new_tokens { max_new = v; }
        if let Some(v) = cfg.system_prompt { system_prompt = Some(v); }
    }

    (temp, k, max_new, system_prompt)
}

// Prepend a system prompt if provided.
fn prepend_system_prompt(prompt: &str, system: Option<&str>) -> String {
    match system {
        Some(s) if !s.trim().is_empty() => format!("{s}\n\n{prompt}"),
        _ => prompt.to_string(),
    }
}

// Interactive REPL loop that keeps conversation history.
fn run_repl(
    model: &Gpt<BackendImpl>,
    cfg: &crate::config::ModelConfig,
    memory_dir: Option<&str>,
    mut store_opt: Option<MemoryStore>,
    mut embedder_opt: Option<HashEmbedder>,
    k_episodic: usize,
    k_semantic: usize,
    wm_words: usize,
    consolidate: bool,
    system_prompt: Option<String>,
    temperature: f64,
    top_k: usize,
    max_new_tokens: usize,
) -> Result<()> {
    use std::io::{self, Write};

    let mut history = String::new();
    let mut sources: Vec<String> = Vec::new();

    loop {
        print!("you> ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("q") {
            break;
        }

        sources.clear();
        let mut prompt = if history.is_empty() {
            format!("User: {input}\nAssistant:")
        } else {
            format!("{history}\nUser: {input}\nAssistant:")
        };

        if let (Some(_memdir), Some(store), Some(embedder)) = (memory_dir, store_opt.as_ref(), embedder_opt.as_ref()) {
            let q_emb = embedder.embed(input);
            let epi = top_k_chunks(&q_emb, &store.chunks, k_episodic);
            let sem = top_k_notes(&q_emb, &store.notes, k_semantic);

            let episodic_lines: Vec<String> = epi.iter().map(|(c, _s)| {
                sources.push(c.id.clone());
                format!("[{}:{}] {}", c.book, c.offset, trim_text(&c.text, 60))
            }).collect();
            let semantic_lines: Vec<String> = sem.iter().map(|(n, _s)| n.text.clone()).collect();

            prompt = build_working_memory_prompt(&prompt, &semantic_lines, &episodic_lines, wm_words);
        }

        let prompt = prepend_system_prompt(&prompt, system_prompt.as_deref());
        let out = generate(model, cfg, &prompt, max_new_tokens, temperature, top_k);
        println!("assistant> {out}");

        history.push_str(&format!("User: {input}\nAssistant: {out}\n"));

        if consolidate {
            if let (Some(mut store), Some(embedder), Some(memdir)) = (store_opt.take(), embedder_opt.take(), memory_dir) {
                let evidence_chunks: Vec<EvidenceChunk> = sources.iter().filter_map(|id| {
                    store.chunks.iter().find(|c| c.id == *id).map(|c| EvidenceChunk {
                        id: c.id.clone(),
                        text: c.text.clone(),
                    })
                }).collect();

                let new_notes = consolidate_semantic_notes(
                    &embedder,
                    input,
                    &out,
                    &evidence_chunks,
                    &store.notes,
                    store.notes.len(),
                );

                if !new_notes.is_empty() {
                    store.notes.extend(new_notes);
                    store.save_notes(memdir)?;
                }
                store_opt = Some(store);
                embedder_opt = Some(embedder);
            }
        }
    }

    Ok(())
}
