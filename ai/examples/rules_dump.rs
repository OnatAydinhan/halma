//! T-3 cross-check harness: drive the *stateful* rule layer (`apply_move`,
//! `end_turn`, `check_win`, the chain-revisit ban, `_check_win`) and emit a
//! canonical one-line state after each step, so a Python harness can compare it
//! byte-for-byte against `halma8.Halma8`.
//!
//! A single persistent board is maintained across commands. One command per
//! line, one reply per line, in order.
//!
//! State-setup commands (reply = canonical `<state>` line):
//!   `NEW`                 -> reset to `Board::start()` (halma8.Halma8())
//!   `MAKES <board_str>`   -> fresh board whose grid is `board_str`
//!                            (turn=P1, winner=None, turn_no=1, chain off; i.e. halma8 make_board)
//!   `TURN <1|2>`          -> set the turn
//!   `WINNER <0|1|2>`      -> set the winner (0 = None)
//!   `PLACE <r c> <p>`     -> put a piece at (r, c)  (p in {1,2})
//!   `CHECKWIN`            -> run the win check (halma8 `_check_win`)
//!   `ENDTURN`             -> run `end_turn`
//!   `STATE`               -> no-op; emit the current state
//!
//! Move-application commands (reply = `<result> <state>` where result ∈ {T,F,E}):
//!   `STEP <r c> <nr nc>`     -> apply a step
//!   `JUMP <r c> <mr mc> <nr nc>` -> apply a jump
//!
//!   T = `Ok(true)`  (jump, chain still live, turn unchanged)
//!   F = `Ok(false)` (turn ended)
//!   E = `Err(Ill)`  (rejected; the board is left unchanged on both engines)
//!
//! `<state>` = `Board::state_line()` = `<position_str> turn_no=<n> winner=<0|1|2>`.
//!
//! On any parse error: `ERR <message>` (the harness treats any `ERR` line as a
//! hard failure). Used by `crosscheck_rules.py`.

use std::io::{BufRead, Write};

use halma8_ai::engine::{Board, IllegalMove, Move};

fn parse_u8(t: &str) -> Option<u8> {
    let b = t.as_bytes();
    if b.len() != 1 || !(b[0].is_ascii_digit()) {
        return None;
    }
    Some((b[0] - b'0') as u8)
}

fn main() {
    let mut board = Board::start();
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

        // Apply the command to the persistent board; build the raw payload.
        let payload = match cmd {
            "NEW" => {
                board = Board::start();
                board.state_line()
            }
            "MAKES" => match (parts.get(1), Board::from_board_str(parts.get(1).copied().unwrap_or(""))) {
                (Some(bs), Ok(b)) => {
                    board = b;
                    board.state_line()
                }
                _ => format!("ERR bad MAKES: {line}"),
            },
            "TURN" => match parts.get(1).and_then(|t| parse_u8(t)) {
                Some(t) => {
                    board.turn = t;
                    board.state_line()
                }
                None => "ERR TURN needs <1|2>".to_string(),
            },
            "WINNER" => match parts.get(1).and_then(|t| parse_u8(t)) {
                Some(v) => {
                    board.winner = if v == 0 { None } else { Some(v) };
                    board.state_line()
                }
                None => "ERR WINNER needs <0|1|2>".to_string(),
            },
            "PLACE" => match (
                parts.get(1).copied().and_then(parse_u8),
                parts.get(2).copied().and_then(parse_u8),
                parts.get(3).copied().and_then(parse_u8),
            ) {
                (Some(r), Some(c), Some(p)) => {
                    board.grid[r as usize * 8 + c as usize] = p;
                    board.state_line()
                }
                _ => "ERR PLACE needs <r c p>".to_string(),
            }
            "CHECKWIN" => {
                board.check_win();
                board.state_line()
            }
            "ENDTURN" => {
                board.end_turn();
                board.state_line()
            }
            "STATE" => board.state_line(),
            "STEP" => match (
                parts.get(1).copied().and_then(parse_u8),
                parts.get(2).copied().and_then(parse_u8),
                parts.get(3).copied().and_then(parse_u8),
                parts.get(4).copied().and_then(parse_u8),
            ) {
                (Some(r), Some(c), Some(nr), Some(nc)) => {
                    let res = board.apply_move(&Move::Step { from: (r, c), to: (nr, nc) });
                    format!("{} {}", match res {
                        Ok(true) => "T",
                        Ok(false) => "F",
                        Err(Ill) => "E",
                    }, board.state_line())
                }
                _ => "ERR STEP needs <r c> <nr nc>".to_string(),
            },
            "JUMP" => match (
                parts.get(1).copied().and_then(parse_u8),
                parts.get(2).copied().and_then(parse_u8),
                parts.get(3).copied().and_then(parse_u8),
                parts.get(4).copied().and_then(parse_u8),
                parts.get(5).copied().and_then(parse_u8),
                parts.get(6).copied().and_then(parse_u8),
            ) {
                (Some(r), Some(c), Some(mr), Some(mc), Some(nr), Some(nc)) => {
                    let res = board.apply_move(&Move::Jump { from: (r, c), mid: (mr, mc), to: (nr, nc) });
                    format!("{} {}", match res {
                        Ok(true) => "T",
                        Ok(false) => "F",
                        Err(Ill) => "E",
                    }, board.state_line())
                }
                _ => "ERR JUMP needs <r c> <mr mc> <nr nc>".to_string(),
            },
            _ => format!("ERR bad command: {line}"),
        };

        let _ = writeln!(w, "{payload}");
    }
    let _ = w.flush();
}
