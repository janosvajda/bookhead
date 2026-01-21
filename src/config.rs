use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab: usize,      // 256 (byte-level)
    pub seq_len: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

impl ModelConfig {
    // Convenience constructor with sensible defaults for vocab/dropout.
    #[allow(dead_code)]
    pub fn new(seq_len: usize, n_layers: usize, n_heads: usize, d_model: usize, d_ff: usize) -> Self {
        Self {
            vocab: 256,
            seq_len,
            n_layers,
            n_heads,
            d_model,
            d_ff,
            dropout: 0.0,
        }
    }
}
