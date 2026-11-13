//! T-10 round-trip harness: print `evaluate_with(&board, &weights)` for a given
//! position + weights JSON.
//!
//! Protocol (one command per stdin line):
//!   `E <weights_json_path> <position_string>`  →  `OK <score>`
//!   (pass a non-existent path or the literal `default` to use Rust defaults)
//! Errors → `ERR <reason>` (continue to the next line).
//!
//! Used by `ai/crosscheck_eval.py` to verify that the Python mirror of
//! `evaluate.rs::evaluate_with` matches Rust exactly (±1 cp).

use std::io::{BufRead, Write};

use halma8_ai::engine::Board;
use halma8_ai::evaluate::{evaluate_with, Weights};

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l.trim_end().to_string(),
            Err(_) => break,
        };
        if line.is_empty() {
            continue;
        }

        let result: String = match line.strip_prefix("E ") {
            Some(rest) => {
                let mut it = rest.splitn(2, ' ');
                let path = match it.next() {
                    Some(p) if p != "default" => p.to_string(),
                    _ => "__default_weights__".to_string(),
                };
                let pos = it.next().unwrap_or("").to_string();
                match Board::from_position_str(&pos) {
                    Ok(b) => {
                        let w: Weights = if path == "__default_weights__" {
                            Weights::default()
                        } else {
                            Weights::load(&path)
                        };
                        format!("OK {}", evaluate_with(&b, &w))
                    }
                    Err(_) => "ERR bad position string".to_string(),
                }
            }
            None => format!("ERR unknown command: {}", line),
        };

        let _ = writeln!(out, "{}", result);
        let _ = out.flush();
    }
}
