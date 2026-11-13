//! T-2 cross-check harness: set a board, then query `legal_steps` / `legal_jumps`.
//!
//! Commands (one per line, one reply line per command, in order):
//!   `B <board_str>` -> `OK`                  (set the board that later `L` queries use)
//!   `L <r> <c>`     -> `OK steps=<...> jumps=<...>`
//!   `S`             -> `OK <64-char board string of Board::start()>`
//!
//! Canonical serialization (matches `crosscheck_moves.py` on the Python side):
//!   `steps=`  : `|`-joined `<r><c>` tokens, ascending by (r, c); `` if none.
//!   `jumps=`  : `|`-joined `<mr><mc><nr><nc>` tokens, ascending; `` if none.
//!
//! On any parse error: `ERR <message>` (and continue to the next line).
//!
//! Used by `crosscheck_moves.py` to compare Rust move generation against
//! `halma8.Halma8` (the canonical oracle).

use std::io::{BufRead, Write};

use halma8_ai::engine::Board;

fn format_steps(steps: &[(u8, u8)]) -> String {
    let mut v = steps.to_vec();
    v.sort_unstable();
    v.iter().map(|&(r, c)| format!("{r}{c}")).collect::<Vec<_>>().join("|")
}

fn format_jumps(jumps: &[((u8, u8), (u8, u8))]) -> String {
    let mut v = jumps.to_vec();
    v.sort_unstable();
    v.iter()
        .map(|&((mr, mc), (nr, nc))| format!("{mr}{mc}{nr}{nc}"))
        .collect::<Vec<_>>()
        .join("|")
}

fn main() {
    let mut board = Board::new();
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

        let parts: Vec<&str> = line.split_whitespace().collect();
        let cmd = parts.first().copied().unwrap_or("");
        let out_line: String = match cmd {
            "S" => format!("OK {}", Board::start().to_board_str()),
            "B" => {
                let arg = parts.get(1).copied().unwrap_or("");
                match Board::from_board_str(arg) {
                    Ok(b) => {
                        board = b;
                        "OK".to_string()
                    }
                    Err(e) => format!("ERR {e}"),
                }
            }
            "L" => match (parts.get(1), parts.get(2)) {
                (Some(rs), Some(cs)) => match (rs.parse::<u8>(), cs.parse::<u8>()) {
                    (Ok(r), Ok(c)) => {
                        let steps = board.legal_steps(r, c);
                        let jumps = board.legal_jumps(r, c);
                        format!(
                            "OK steps={} jumps={}",
                            format_steps(&steps),
                            format_jumps(&jumps)
                        )
                    }
                    _ => "ERR L needs <r> <c>".to_string(),
                },
                _ => "ERR L needs <r> <c>".to_string(),
            },
            _ => format!("ERR bad command: {line}"),
        };

        let _ = writeln!(w, "{out_line}");
    }
    let _ = w.flush();
}
