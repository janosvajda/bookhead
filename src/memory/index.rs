use anyhow::Result;
use std::fs;
use std::path::Path;

use crate::data::books::read_books_folder;
use crate::memory::embed::HashEmbedder;
use crate::memory::types::Chunk;

// Build episodic and semantic memory stores from book text.
pub fn run_index(books_dir: &str, out_dir: &str, chunk_words: usize, overlap_words: usize, embed_dim: usize) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    let mem_dir = Path::new(out_dir).join("memory");
    fs::create_dir_all(&mem_dir)?;

    let embedder = HashEmbedder::new(embed_dim);

    let books = read_books_folder(books_dir)?;
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut id_counter = 0usize;

    for (book_name, text) in books {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut i = 0usize;
        while i < words.len() {
            let end = (i + chunk_words).min(words.len());
            let chunk_text = words[i..end].join(" ");
            let emb = embedder.embed(&chunk_text);

            let c = Chunk {
                id: format!("c{:08}", id_counter),
                book: book_name.clone(),
                offset: i,
                text: chunk_text,
                embedding: emb,
                importance: 0.1,
            };
            chunks.push(c);
            id_counter += 1;

            if end == words.len() { break; }
            i = end.saturating_sub(overlap_words);
        }
    }

    // Write episodic.jsonl
    let episodic_path = mem_dir.join("episodic.jsonl");
    let mut out = String::new();
    for c in &chunks {
        out.push_str(&serde_json::to_string(c)?);
        out.push('\n');
    }
    fs::write(episodic_path, out)?;

    // Initialize empty semantic.jsonl
    let semantic_path = mem_dir.join("semantic.jsonl");
    if !semantic_path.exists() {
        fs::write(semantic_path, "")?;
    }

    // Save embedder config for reproducibility
    fs::write(mem_dir.join("embedder.json"), format!(r#"{{"type":"hash","dim":{}}}"#, embed_dim))?;

    println!("Indexed {} chunks into {:?}", chunks.len(), mem_dir);
    Ok(())
}
