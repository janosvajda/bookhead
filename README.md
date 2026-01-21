# Experimental LLM with short and long term memory

This project includes BOTH:

1) A **decoder-only Transformer (GPT-style)** model + training CLI (short-term memory = context window).
2) A **long-term memory system** (episodic + semantic) that is retrieved and injected into the prompt (working memory).

## What you asked for: short + long term memory combined

- **Short-term memory**: the model context window (`--seq-len` bytes/tokens).
- **Long-term episodic memory**: book chunks stored in `memory/episodic.jsonl`.
- **Long-term semantic memory**: compressed notes stored in `memory/semantic.jsonl`.
- **Combination**: `chat` retrieves top-k episodic + semantic items and builds a *working memory prompt* that fits inside a budget (`--wm-words`).


## Tests

```bash
cargo test
```

Smoke test (end-to-end):

```bash
cargo test --test smoke
```

Loss-decrease test (ignored by default because it can be flaky across hardware/backends):

```bash
cargo test --test smoke -- --ignored
```

## Build

```bash
cargo build --release
```

You can build and run in one step:

```bash
cargo run --release -- train --books ./books --out ./out_model --workers 8
```

Or build once, then run the binary directly:

```bash
cargo build --release
./target/release/bookhead train --books ./books --out ./out_model --workers 8
```

## 1) Index books into long-term memory

Put your books as `.txt` files in `./books`, then:

```bash
cargo run --release -- index --books ./books --out ./out
```

This creates:
- `./out/memory/episodic.jsonl`  (chunks)
- `./out/memory/semantic.jsonl`  (notes; initially empty)
- `./out/memory/embedder.json`   (embedder config)

## 2) Train the model (optional but included)

```bash
cargo run --release -- train --books ./books --out ./out_model --seq-len 256 --batch 8 --steps 2000
```

Writes `./out_model/config.json` and Burn checkpoints.

### Training speed tips

- **Confirm GPU vs CPU**: when training starts, it prints a line like `Training backend: ...`.
  - `Training backend: ndarray CPU` means training is running only on the processor (CPU). It uses Burn's CPU backend (named `ndarray`) and is much slower.
  - `Training backend: wgpu GPU (...)` means training is running on the graphics card (GPU) and is much faster.
- **Force Metal on macOS** (if GPU is available):
  ```bash
  WGPU_BACKEND=metal WGPU_POWER_PREF=high cargo run --release -- train --books ./books --out ./out_model --seq-len 256 --batch 8 --steps 2000
  ```
- **Use a smaller model while iterating** (much faster):
  ```bash
  cargo run --release -- train --books ./books --out ./out_model --seq-len 128 --batch 8 --steps 2000 \
    --d-model 128 --n-layers 4 --n-heads 4 --d-ff 512
  ```
- **Lower steps for quick tests**: `--steps` is the number of batches, not epochs.
- **Increase dataloader workers**: `--workers 8` (or higher if you have CPU cores).
- **Avoid rebuild overhead**: build once, then run the binary:
  ```bash
  cargo build --release
  ./target/release/book_llm_memory train --books ./books --out ./out_model --workers 8
  ```

Optional settings (early stop, etc.):

```bash
cargo run --release -- train --books ./books --out ./out_model --settings ./settings.json
```

See `settings.json.example` for available fields.

## 3) Chat using both memories

```bash
cargo run --release -- chat --ckpt ./out_model --memory ./out/memory --prompt "Why did the hero leave the village?"
```

Interactive REPL:

```bash
cargo run --release -- chat --ckpt ./out_model --memory ./out/memory --repl
```

Tune retrieval:
- `--k-episodic 5`
- `--k-semantic 5`
- `--wm-words 800`

### Consolidation (semantic memory grows)
Add `--consolidate` to append a semantic note (Q/A) into `semantic.jsonl`:

```bash
cargo run --release -- chat --ckpt ./out_model --memory ./out/memory --prompt "..." --consolidate
```

## Model weight loading

`chat` will attempt to load the latest Burn checkpoint from `./out_model/checkpoint/model-*.{ext}` if present.
If no checkpoint is found, it falls back to an untrained model initialized from `config.json`.
