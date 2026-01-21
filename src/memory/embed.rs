// Rust-only embedding for retrieval: signed feature hashing (words + char n-grams).
// This is NOT the LLM; it's for long-term memory search.

pub struct HashEmbedder {
    pub dim: usize,
}

impl HashEmbedder {
    // Create a hashing embedder with the given vector size.
    pub fn new(dim: usize) -> Self { Self { dim } }

    // Generate a normalized hash-based embedding for a text.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];

        // word hashing (signed feature hashing reduces collisions)
        for tok in text.split_whitespace() {
            add_feature(&mut v, tok.as_bytes(), 1.0);
        }

        // char n-gram hashing (helps for names and partial matches)
        let bytes = text.as_bytes();
        if bytes.len() >= 3 {
            for n in 3..=5 {
                if bytes.len() < n {
                    break;
                }
                for i in 0..=(bytes.len() - n) {
                    add_feature(&mut v, &bytes[i..i + n], 0.3);
                }
            }
        }

        // L2 normalize so cosine similarity is stable.
        let norm = v.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-8);
        for x in &mut v { *x /= norm; }
        v
    }
}

// Add a signed feature hash into the dense vector.
fn add_feature(v: &mut [f32], data: &[u8], weight: f32) {
    let idx = hash_index(data, v.len());
    let sign = hash_sign(data);
    v[idx] += weight * sign;
}

// Standard FNV-1a constants: starting offset and per-byte multiplier.
const FNV_OFFSET: u64 = 1469598103934665603;
const FNV_PRIME: u64 = 1099511628211;

// FNV-1a hash for stable feature indexing.
fn fnv1a_u64(data: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// Map a feature to an index in the vector.
fn hash_index(data: &[u8], dim: usize) -> usize {
    (fnv1a_u64(data, FNV_OFFSET) as usize) % dim
}

// Use a second hash to pick a sign for collision cancellation.
fn hash_sign(data: &[u8]) -> f32 {
    let h = fnv1a_u64(data, FNV_OFFSET ^ 0x9e3779b97f4a7c15);
    if (h & 1) == 0 { 1.0 } else { -1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_is_deterministic_and_normalized() {
        let embedder = HashEmbedder::new(64);
        let v1 = embedder.embed("Hello world");
        let v2 = embedder.embed("Hello world");
        assert_eq!(v1.len(), 64);
        assert_eq!(v1, v2);
        let norm = v1.iter().map(|x| x*x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
