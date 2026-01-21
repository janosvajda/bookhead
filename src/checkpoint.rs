use anyhow::{Result, Context};
use std::fs;
use std::path::Path;

use crate::config::ModelConfig;

// Persist model config as pretty JSON in the output directory.
pub fn save_config(out_dir: &str, cfg: &ModelConfig) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    let path = Path::new(out_dir).join("config.json");
    let json = serde_json::to_string_pretty(cfg)?;
    fs::write(path, json)?;
    Ok(())
}

// Load model config from a JSON file in the given directory.
pub fn load_config(dir: &str) -> Result<ModelConfig> {
    let path = Path::new(dir).join("config.json");
    let s = fs::read_to_string(&path).with_context(|| format!("Failed to read {:?}", path))?;
    Ok(serde_json::from_str(&s)?)
}
