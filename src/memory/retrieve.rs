use crate::memory::types::{Chunk, SemanticNote};

// Compute cosine similarity between two vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt().max(1e-8) * nb.sqrt().max(1e-8))
}

fn top_k_by<T>(items: &[T], k: usize, mut score: impl FnMut(&T) -> f32) -> Vec<(&T, f32)> {
    if k == 0 || items.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(&T, f32)> = items.iter().map(|item| (item, score(item))).collect();
    let keep = k.min(scored.len());
    scored.select_nth_unstable_by(keep - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top = &mut scored[..keep];
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top.to_vec()
}

// Return top-k episodic chunks by cosine similarity.
pub fn top_k_chunks<'a>(q: &[f32], chunks: &'a [Chunk], k: usize) -> Vec<(&'a Chunk, f32)> {
    top_k_by(chunks, k, |c| cosine(q, &c.embedding))
}

// Return top-k semantic notes by cosine similarity.
pub fn top_k_notes<'a>(q: &[f32], notes: &'a [SemanticNote], k: usize) -> Vec<(&'a SemanticNote, f32)> {
    top_k_by(notes, k, |n| cosine(q, &n.embedding))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{Chunk, SemanticNote};

    #[test]
    fn top_k_returns_highest_scores() {
        let q = vec![1.0, 0.0];
        let chunks = vec![
            Chunk { id: "a".to_string(), book: "b".to_string(), offset: 0, text: "a".to_string(), embedding: vec![1.0, 0.0], importance: 0.0 },
            Chunk { id: "b".to_string(), book: "b".to_string(), offset: 1, text: "b".to_string(), embedding: vec![0.0, 1.0], importance: 0.0 },
            Chunk { id: "c".to_string(), book: "b".to_string(), offset: 2, text: "c".to_string(), embedding: vec![0.5, 0.5], importance: 0.0 },
        ];
        let top = top_k_chunks(&q, &chunks, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0.id, "a");

        let notes = vec![
            SemanticNote { id: "n1".to_string(), text: "n1".to_string(), embedding: vec![1.0, 0.0], confidence: 0.0, sources: vec![] },
            SemanticNote { id: "n2".to_string(), text: "n2".to_string(), embedding: vec![0.0, 1.0], confidence: 0.0, sources: vec![] },
        ];
        let top_notes = top_k_notes(&q, &notes, 1);
        assert_eq!(top_notes.len(), 1);
        assert_eq!(top_notes[0].0.id, "n1");
    }
}
