use crate::memory::embed::HashEmbedder;
use crate::memory::types::SemanticNote;
use crate::memory::retrieve::cosine;

#[derive(Debug, Clone)]
pub struct EvidenceChunk {
    pub id: String,
    pub text: String,
}

/// Proper consolidation:
/// - Extract fact-like sentences from the answer
/// - Require evidence support from retrieved episodic memory
/// - Deduplicate against existing semantic notes
/// - Assign confidence based on evidence strength
pub fn consolidate_semantic_notes(
    embedder: &HashEmbedder,
    question: &str,
    answer: &str,
    evidence: &[EvidenceChunk],
    existing_notes: &[SemanticNote],
    next_id_start: usize,
) -> Vec<SemanticNote> {
    let mut out: Vec<SemanticNote> = Vec::new();
    let mut next_id = next_id_start;

    let candidates = extract_candidate_facts(answer);
    if candidates.is_empty() {
        return out;
    }

    for cand in candidates {
        // Compute evidence support
        let (support, sources) = evidence_support(&cand, evidence);
        if support < 0.18 || sources.is_empty() {
            continue; // don't store ungrounded facts
        }

        // Create normalized semantic text
        let text = normalize_fact(question, &cand);

        // Deduplicate against existing notes + notes we just created
        let emb = embedder.embed(&text);
        if is_duplicate(&emb, existing_notes, &out) {
            continue;
        }

        let confidence = (0.2 + 0.8 * support).clamp(0.0, 1.0);

        out.push(SemanticNote {
            id: format!("s{:08}", next_id),
            text,
            embedding: emb,
            confidence,
            sources,
        });

        next_id += 1;
    }

    out
}

/// Extract fact-like sentences:
/// - avoid very short fillers
/// - avoid questions
/// - avoid pure emotion-only lines
fn extract_candidate_facts(answer: &str) -> Vec<String> {
    let mut facts = Vec::new();

    for s in split_sentences(answer) {
        let s = s.trim();
        if s.len() < 35 { continue; }
        if s.ends_with('?') { continue; }

        // filter common chatty fillers
        let lower = s.to_lowercase();
        let filler = [
            "i think", "maybe", "it seems", "as an ai", "i'm not sure",
            "i cannot", "sorry", "thank you", "i understand",
        ];
        if filler.iter().any(|f| lower.starts_with(f)) {
            continue;
        }

        // keep if it contains a "because"/causal/statement vibe or named-ish tokens
        let has_structure = lower.contains(" because ")
            || lower.contains(" so ")
            || lower.contains(" therefore ")
            || lower.contains(" due to ")
            || count_capitalized_tokens(s) >= 1;

        if has_structure {
            facts.push(s.to_string());
        }
    }

    // limit to avoid memory spam
    facts.truncate(5);
    facts
}

/// Evidence support: keyword overlap between candidate and evidence chunks.
/// Returns (support_score, sources_chunk_ids)
fn evidence_support(candidate: &str, evidence: &[EvidenceChunk]) -> (f32, Vec<String>) {
    let cand_keys = keywords(candidate);
    if cand_keys.is_empty() {
        return (0.0, vec![]);
    }

    let mut best = 0.0f32;
    let mut sources = Vec::new();

    for e in evidence {
        let e_keys = keywords(&e.text);
        if e_keys.is_empty() { continue; }

        let overlap = jaccard(&cand_keys, &e_keys);
        // give a small bonus if the evidence contains "because/due to" too
        let bonus = if e.text.to_lowercase().contains("because") { 0.03 } else { 0.0 };
        let score = (overlap as f32) + bonus;

        if score > 0.12 {
            sources.push(e.id.clone());
        }
        if score > best {
            best = score;
        }
    }

    // Normalize best into 0..1-ish range
    let support = (best / 0.45).clamp(0.0, 1.0);
    (support, sources)
}

/// Create a compact semantic memory item.
/// Keeps it short and reusable.
fn normalize_fact(question: &str, fact: &str) -> String {
    // Keep semantic notes compact and declarative
    format!("Fact (from Q: {}): {}", squash_whitespace(question), squash_whitespace(fact))
}

fn is_duplicate(emb: &[f32], existing: &[SemanticNote], newly: &[SemanticNote]) -> bool {
    for n in existing {
        if cosine(emb, &n.embedding) > 0.92 {
            return true;
        }
    }
    for n in newly {
        if cosine(emb, &n.embedding) > 0.92 {
            return true;
        }
    }
    false
}

fn split_sentences(text: &str) -> Vec<String> {
    // simple splitter (no external crates)
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        buf.push(ch);
        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            let s = buf.trim().to_string();
            if !s.is_empty() {
                out.push(s);
            }
            buf.clear();
        }
    }
    let tail = buf.trim().to_string();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn keywords(text: &str) -> Vec<String> {
    // lowercase, strip punctuation, remove stopwords, keep 4..20 char tokens
    let stop = [
        "the","a","an","and","or","but","to","of","in","on","for","with","as","at","by",
        "is","are","was","were","be","been","being","it","this","that","these","those",
        "i","you","he","she","they","we","me","him","her","them","my","your","their","our",
        "not","just","very","really","so","because","due"
    ];

    let mut out = Vec::new();
    let cleaned: String = text.chars().map(|c| {
        if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }
    }).collect();

    for tok in cleaned.split_whitespace() {
        let t = tok.to_lowercase();
        if t.len() < 4 || t.len() > 20 { continue; }
        if stop.contains(&t.as_str()) { continue; }
        out.push(t);
    }

    out.sort();
    out.dedup();
    out
}

fn jaccard(a: &[String], b: &[String]) -> f32 {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut inter = 0usize;

    while i < a.len() && j < b.len() {
        if a[i] == b[j] { inter += 1; i += 1; j += 1; }
        else if a[i] < b[j] { i += 1; }
        else { j += 1; }
    }

    let union = a.len() + b.len() - inter;
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}

fn squash_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn count_capitalized_tokens(s: &str) -> usize {
    s.split_whitespace()
        .filter(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
        .count()
}
