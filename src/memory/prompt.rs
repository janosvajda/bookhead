// Assemble a prompt with semantic/episodic memory under a word budget.
pub fn build_working_memory_prompt(
    user_prompt: &str,
    semantic: &[String],
    episodic: &[String],
    wm_words: usize,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    parts.push("You are an assistant answering questions about the books.\nRules:\n- Use ONLY the provided memories.\n- If the memory is insufficient, say you don't know.\n".to_string());

    if !semantic.is_empty() {
        parts.push("Long-term semantic memory (compressed notes):".to_string());
        for s in semantic {
            parts.push(format!("- {}", s));
        }
        parts.push("".to_string());
    }

    if !episodic.is_empty() {
        parts.push("Long-term episodic memory (book excerpts):".to_string());
        for e in episodic {
            parts.push(format!("- {}", e));
        }
        parts.push("".to_string());
    }

    parts.push(format!("User: {}", user_prompt));
    parts.push("Assistant:".to_string());

    // crude word-budget truncation
    let mut out = String::new();
    let mut used = 0usize;
    for p in parts {
        let w = p.split_whitespace().count();
        if used + w > wm_words { break; }
        used += w;
        out.push_str(&p);
        out.push('\n');
    }
    out
}
