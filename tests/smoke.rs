use std::{env, fs, path::PathBuf, process::Command, time::{SystemTime, UNIX_EPOCH}};

fn make_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = env::temp_dir().join(format!("bookhead_{prefix}_{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn resolve_bin() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_bookhead") {
        return PathBuf::from(path);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push("debug");
    let bin_name = if cfg!(windows) { "bookhead.exe" } else { "bookhead" };
    path.push(bin_name);
    if !path.is_file() {
        let status = Command::new("cargo")
            .args(["build", "--bin", "bookhead"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status()
            .expect("cargo build");
        assert!(status.success(), "cargo build failed");
    }
    path
}

#[test]
fn train_and_chat_smoke() {
    let bin = resolve_bin();
    let books_dir = make_temp_dir("books");
    let out_dir = make_temp_dir("out");

    fs::write(books_dir.join("book.txt"), "Hello world. This is a tiny book.").unwrap();

    let status = Command::new(&bin)
        .args([
            "train",
            "--books", books_dir.to_str().unwrap(),
            "--out", out_dir.to_str().unwrap(),
            "--seq-len", "16",
            "--batch", "2",
            "--steps", "1",
            "--d-model", "32",
            "--n-layers", "2",
            "--n-heads", "2",
            "--d-ff", "64",
            "--workers", "1",
        ])
        .status()
        .expect("train command");
    assert!(status.success(), "train failed");

    let config_path = out_dir.join("config.json");
    assert!(config_path.is_file(), "missing config.json");

    let checkpoint_dir = out_dir.join("checkpoint");
    let _ = checkpoint_dir;

    let output = Command::new(&bin)
        .args([
            "chat",
            "--ckpt", out_dir.to_str().unwrap(),
            "--prompt", "Hello",
            "--max-new-tokens", "4",
            "--temperature", "0.0",
            "--top-k", "1",
        ])
        .output()
        .expect("chat command");
    assert!(output.status.success(), "chat failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.trim().is_empty(), "empty chat output");

    let _ = fs::remove_dir_all(&books_dir);
    let _ = fs::remove_dir_all(&out_dir);
}

#[test]
#[ignore]
fn loss_decreases_short_run() {
    let bin = resolve_bin();
    let books_dir = make_temp_dir("books_loss");
    let out_dir = make_temp_dir("out_loss");
    let loss_path = make_temp_dir("loss").join("loss.txt");

    fs::write(books_dir.join("book.txt"), "aaaaa aaaaa aaaaa aaaaa aaaaa").unwrap();

    let status = Command::new(&bin)
        .env("BOOKHEAD_LOSS_LOG", loss_path.to_str().unwrap())
        .args([
            "train",
            "--books", books_dir.to_str().unwrap(),
            "--out", out_dir.to_str().unwrap(),
            "--seq-len", "8",
            "--batch", "2",
            "--steps", "8",
            "--d-model", "32",
            "--n-layers", "2",
            "--n-heads", "2",
            "--d-ff", "64",
            "--workers", "1",
        ])
        .status()
        .expect("train command");
    assert!(status.success(), "train failed");

    let content = fs::read_to_string(&loss_path).expect("loss log");
    let mut first = None;
    let mut last = None;
    for line in content.lines() {
        if let Some(v) = line.strip_prefix("first=") {
            first = v.parse::<f64>().ok();
        }
        if let Some(v) = line.strip_prefix("last=") {
            last = v.parse::<f64>().ok();
        }
    }
    let first = first.expect("first loss");
    let last = last.expect("last loss");
    assert!(last < first, "loss did not decrease: {first} -> {last}");

    let _ = fs::remove_dir_all(&books_dir);
    let _ = fs::remove_dir_all(&out_dir);
}
