mod config;
mod data;
mod model;
mod train;
mod infer;
mod checkpoint;
mod memory;
mod settings;

use clap::{Parser, Subcommand};
use anyhow::Result;

#[derive(Parser)]
#[command(name = "bookhead")]
#[command(about = "Train a GPT-style decoder-only Transformer on books, build memory index, and chat with short+long term memory.", long_about=None)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a model from scratch on a folder of .txt files
    Train {
        #[arg(long)]
        books: String,
        #[arg(long)]
        out: String,
        #[arg(long, default_value_t = 256)]
        seq_len: usize,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        #[arg(long, default_value_t = 256)]
        d_model: usize,
        #[arg(long, default_value_t = 8)]
        n_layers: usize,
        #[arg(long, default_value_t = 8)]
        n_heads: usize,
        #[arg(long, default_value_t = 1024)]
        d_ff: usize,
        #[arg(long, default_value_t = 200)]
        save_every: usize,
        /// Number of dataloader workers (0/1 = single-threaded)
        #[arg(long, default_value_t = 4)]
        workers: usize,
        /// Optional JSON settings file (early stop, etc.)
        #[arg(long)]
        settings: Option<String>,
    },

    /// Build long-term memory index (episodic chunks + embeddings) from books
    Index {
        #[arg(long)]
        books: String,
        #[arg(long)]
        out: String,
        /// words per chunk
        #[arg(long, default_value_t = 220)]
        chunk_words: usize,
        /// overlap between chunks
        #[arg(long, default_value_t = 60)]
        overlap_words: usize,
        /// embedding dim for hashing embedder
        #[arg(long, default_value_t = 2048)]
        embed_dim: usize,
    },

    /// Chat with a trained checkpoint and memory store (short+long-term memory)
    Chat {
        #[arg(long)]
        ckpt: String,
        /// Memory directory (created by Index). If omitted, chat is model-only.
        #[arg(long)]
        memory: Option<String>,
        #[arg(long)]
        prompt: Option<String>,
        /// Interactive REPL chat loop (ignores --prompt)
        #[arg(long, default_value_t = false)]
        repl: bool,
        #[arg(long, default_value_t = 200)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// How many episodic chunks to retrieve into working memory
        #[arg(long, default_value_t = 5)]
        k_episodic: usize,
        /// How many semantic notes to retrieve into working memory
        #[arg(long, default_value_t = 5)]
        k_semantic: usize,
        /// Working memory budget (approx words in prompt)
        #[arg(long, default_value_t = 800)]
        wm_words: usize,
        /// If set, write/update semantic memory after answering (simple consolidation)
        #[arg(long, default_value_t = false)]
        consolidate: bool,
        /// Optional JSON settings file (chat tuning, emotional presets)
        #[arg(long)]
        settings: Option<String>,
    },
}

// Parse CLI args and dispatch to subcommands.
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Train { books, out, seq_len, batch, steps, lr, d_model, n_layers, n_heads, d_ff, save_every, workers, settings } => {
            train::run_train(&books, &out, seq_len, batch, steps, lr, d_model, n_layers, n_heads, d_ff, save_every, workers, settings.as_deref())
        }
        Command::Index { books, out, chunk_words, overlap_words, embed_dim } => {
            memory::index::run_index(&books, &out, chunk_words, overlap_words, embed_dim)
        }
        Command::Chat { ckpt, memory, prompt, repl, max_new_tokens, temperature, top_k, k_episodic, k_semantic, wm_words, consolidate, settings } => {
            infer::run_chat_with_memory(
                &ckpt,
                memory.as_deref(),
                prompt.as_deref(),
                repl,
                max_new_tokens,
                temperature,
                top_k,
                k_episodic,
                k_semantic,
                wm_words,
                consolidate,
                settings.as_deref(),
            )
        }
    }
}
