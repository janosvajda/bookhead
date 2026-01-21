use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AppSettings {
    pub training: Option<TrainingSettings>,
    pub chat: Option<ChatSettings>,
}

impl AppSettings {
    // Load app settings from a JSON file.
    pub fn load(path: &str) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| format!("Failed to read {path}"))?;
        Ok(serde_json::from_str(&raw)?)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingSettings {
    pub early_stop: Option<EarlyStopSettings>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EarlyStopSettings {
    pub enabled: bool,
    pub patience: usize,
    pub min_delta: f64,
    pub use_validation: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatSettings {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub max_new_tokens: Option<usize>,
    pub system_prompt: Option<String>,
    pub emotional_preset: Option<String>,
}
