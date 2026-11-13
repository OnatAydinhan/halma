//! T-1 cross-check harness: reads commands from stdin, prints to stdout.
//!
//! Commands (one per line):
//!   `S`          -> `OK <board string of the Rust start position>`
//!   `P <pos>`    -> `OK <position string re-serialized by Rust>`
//!   `B <board>`  -> `OK <board string re-serialized by Rust>`
//!
//! On any error: `ERR <message>` (and continue to the next line).
//!
//! Used by `crosscheck_board.py` to compare the Rust engine's own board
//! state and serialization against `halma8.Halma8` (the canonical oracle).

use std::io::{BufRead, Write};

use halma8_ai::engine::Board;

fn main() {
    let stdin = std::io::stdin();
    let out = std::io::stdout();
    let mut w = out.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l.trim_end().to_string(),
            Err(_) => break,
        };
        if line.is_empty() {
            continue;
        }

        let out_line: String = if line == "S" {
            format!("OK {}", Board::start().to_board_str())
        } else {
            let mut it = line.splitn(2, ' ');
            let cmd = it.next().unwrap_or("");
            let arg = it.next().unwrap_or("").to_string();
            let res = match cmd {
                "B" => Board::from_board_str(&arg).map(|b| b.to_board_str()).ok(),
                "P" => Board::from_position_str(&arg)
                    .map(|b| b.to_position_str())
                    .ok(),
                _ => None,
            };
            match res {
                Some(r) => format!("OK {r}"),
                None => format!("ERR bad command: {}", line),
            }
        };

        let _ = writeln!(w, "{out_line}");
    }
    let _ = w.flush();
}
