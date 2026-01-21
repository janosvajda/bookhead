use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub book: String,
    pub offset: usize,
    pub text: String,
    pub embedding: Vec<f32>,
    pub importance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticNote {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub confidence: f32,
    pub sources: Vec<String>,
}
