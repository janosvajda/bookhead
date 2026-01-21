use anyhow::{Result, Context};
use std::fs;
use std::path::Path;

use crate::memory::types::{Chunk, SemanticNote};

#[derive(Debug, Clone)]
pub struct MemoryStore {
    pub chunks: Vec<Chunk>,
    pub notes: Vec<SemanticNote>,
}

impl MemoryStore {
    // Load episodic chunks and semantic notes from JSONL files.
    pub fn load(dir: &str) -> Result<Self> {
        let chunks_path = Path::new(dir).join("episodic.jsonl");
        let notes_path  = Path::new(dir).join("semantic.jsonl");

        let mut chunks = Vec::new();
        let mut notes = Vec::new();

        let chunks_txt = fs::read_to_string(&chunks_path)
            .with_context(|| format!("Failed to read {:?}", chunks_path))?;
        for line in chunks_txt.lines() {
            if line.trim().is_empty() { continue; }
            chunks.push(serde_json::from_str::<Chunk>(line)?);
        }

        if notes_path.exists() {
            let notes_txt = fs::read_to_string(&notes_path)
                .with_context(|| format!("Failed to read {:?}", notes_path))?;
            for line in notes_txt.lines() {
                if line.trim().is_empty() { continue; }
                notes.push(serde_json::from_str::<SemanticNote>(line)?);
            }
        }

        Ok(Self { chunks, notes })
    }

    // Persist semantic notes back to disk.
    pub fn save_notes(&self, dir: &str) -> Result<()> {
        let notes_path  = Path::new(dir).join("semantic.jsonl");
        let mut out = String::new();
        for n in &self.notes {
            out.push_str(&serde_json::to_string(n)?);
            out.push('\n');
        }
        fs::write(notes_path, out)?;
        Ok(())
    }
}
