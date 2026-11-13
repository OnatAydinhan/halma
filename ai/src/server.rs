//! Stdin/stdout line protocol (UCI-like): `uci` / `isready` / `position` /
//! `go depth|movetime` / `stop` / `setoption` / `quit`.
//! Port of `kego_engine_v2/src/uci.rs` (T-7).
//!
//! Design notes (T-7):
//! - A dedicated reader thread blocks on stdin and forwards each line to the
//!   main (event) thread through an `mpsc` channel. This keeps stdin readable
//!   *while* a search runs, so `go` can be dispatched into a background search
//!   thread without the main thread's line-read being blocked off.
//! - `go` spawns the search in its own thread. The engine's stop flag
//!   (`Search.stop: Arc<AtomicBool>`) is shared with the main thread by cloning
//!   the `Arc` before the search is spawned; on a `stop` line the main thread
//!   atomically sets *that same* flag (the only `Search` field it ever writes),
//!   which reproduces `Search::request_stop` exactly.
//! - While waiting for the search to finish we poll (a) the result channel
//!   and (b) fresh protocol lines in a small bounded loop, so `stop` and
//!   `isready` can be honored even mid-search, and `bestmove` is emitted as
//!   soon as the search thread's result lands — no OS-level `select()` needed,
//!   keeps the crate std-only and portable (Windows included).

use crate::engine::Board;
use crate::evaluate::Weights;
use crate::search::{Play, Search};
use std::io::{BufRead, Write};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;

/// Wire-format string for a search action (matches `bench.rs::play_str` and the
/// plan's "Move string" section: `step r,c nr,nc` / `jump r,c mr,mc nr,nc` / `stop`).
fn play_str(p: &Play) -> String {
    match p {
        Play::Step { from, to } => format!("step {},{} {},{}", from.0, from.1, to.0, to.1),
        Play::Jump { from, mid, to } => format!(
            "jump {},{} {},{} {},{}",
            from.0,
            from.1,
            mid.0,
            mid.1,
            to.0,
            to.1
        ),
        Play::Stop => "stop".to_string(),
    }
}

fn trim_line(line: &str) -> &str {
    line.trim_end_matches(['\n', '\r']).trim()
}

fn emit(out: &mut (impl Write + ?Sized), line: &str) {
    let _ = writeln!(out, "{}", line);
    let _ = out.flush();
}

fn main_loop(out: &mut (impl Write + ?Sized), lines: mpsc::Receiver<String>) {
    use mpsc::TryRecvError;

    let mut board: Option<Board> = None;
    let mut quit_requested = false;
    // T-G0: per-side learned weights for the live search. Both default to
    // `Weights::default()`, so a server that never receives a `setoption` is
    // byte-identical to the pre-T-G0 server (selfplay/bench paths untouched).
    // Seat ↔ side is constant for the whole game and set once before the first
    // `go`: P1 = Red (bottom-left base) → `w1`, P2 = Blue (top-right base) → `w2`.
    let mut w1 = Weights::default();
    let mut w2 = Weights::default();

    while !quit_requested {
        let line = match lines.recv() {
            Ok(l) => l,
            Err(_) => break, // reader thread gone (EOF on stdin)
        };
        if line.trim().is_empty() {
            continue;
        }

        let words: Vec<&str> = line.split_whitespace().collect();
        let cmd = words[0];
        match cmd {
            "uci" => {
                emit(out, "id name check_halma_ai");
                emit(out, "id author check_halma");
                emit(out, "uciok");
            }
            "isready" => {
                emit(out, "readyok");
            }
            "position" => {
                // `position <board> <turn> <chain>` — everything after the first token.
                let pos = line
                    .splitn(2, char::is_whitespace)
                    .nth(1)
                    .map(str::trim)
                    .unwrap_or("");
                match Board::from_position_str(pos) {
                    Ok(b) => board = Some(b),
                    Err(e) => emit(out, &format!("error {}", e)),
                }
            }
            "setoption" => {
                // T-G0: `setoption name <WeightsP1|WeightsP2|Weights> path=<file>`
                //   WeightsP1 → P1 (Red) weights,  WeightsP2 → P2 (Blue) weights,
                //   Weights   → both (convenience for single-model play / analysis).
                // The value token is only honoured when it is exactly `path=<file>`
                // (a `path=` with spaces is out of scope — model paths here have
                // none). Unknown options, and a recognised name with no `path=`,
                // stay silent — UCI's setoption contract is no-error, no-ack.
                let name = words.get(2).copied();
                let path = words.get(3).and_then(|t| t.strip_prefix("path="));
                match (name, path) {
                    (Some("WeightsP1"), Some(p)) => {
                        w1 = Weights::load(p);
                        eprintln!("[weights] P1 <- {}", p);
                    }
                    (Some("WeightsP2"), Some(p)) => {
                        w2 = Weights::load(p);
                        eprintln!("[weights] P2 <- {}", p);
                    }
                    (Some("Weights"), Some(p)) => {
                        let loaded = Weights::load(p);
                        w1 = loaded;
                        w2 = loaded;
                        eprintln!("[weights] both <- {}", p);
                    }
                    _ => {} // unknown option / missing `path=` → silent (UCI)
                }
            }
            "quit" => {
                quit_requested = true;
            }
            "go" => {
                let (movetime, max_depth): (Option<u64>, Option<i32>) =
                    if words.get(1).map(|s| *s) == Some("depth") {
                        (None, words.get(2).and_then(|s| s.parse().ok()))
                    } else if words.get(1).map(|s| *s) == Some("movetime") {
                        (words.get(2).and_then(|s| s.parse().ok()), None)
                    } else {
                        (None, None)
                    };

                let depth_cap = match max_depth {
                    Some(d) => d,
                    None => 64, // movetime mode (or bare `go`): depth is just a generous safety cap
                };

                let pos = board.clone().unwrap_or_else(Board::start);

                let (res_tx, res_rx) = mpsc::channel::<(i32, Option<Play>, i32)>();
                let mut s = Search::new();
                // T-G0: feed the per-side weights captured by `setoption` (both
                // default if none was sent) into this search. Set *before* the
                // search thread is spawned — read-only during search, no shared
                // mutation under the thread.
                s.set_weights_pair(w1, w2);
                if let Some(ms) = movetime {
                    s.set_movetime_ms(ms);
                }
                s.set_max_depth(depth_cap);

                // The stop flag is a shared `Arc<AtomicBool>` (`Search.stop`), so we
                // clone the `Arc` *before* moving `s` into the search thread and then
                // atomically set it from here. Both threads touch the *same* flag — no
                // raw pointer, no reliance on where `s` physically lives after the
                // move. `stop` is the only `Search` field this thread ever writes, so
                // there is no data race on any non-atomic field.
                let stop_flag = s.stop.clone();

                let handle = thread::spawn(move || {
                    let (score, play) = s.search(&pos, depth_cap);
                    let deep = s.deep_completed;
                    let _ = res_tx.send((score, play, deep));
                });

                let mut stop_sent = false;
                loop {
                    if let Ok((score, play, deep)) = res_rx.try_recv() {
                        if let Some(deep) = Some(deep).filter(|_| deep > 0) {
                            emit(out, &format!("info depth {} score cp {}", deep, score));
                        } else {
                            emit(out, &format!("info score cp {}", score));
                        }
                        let mv = match &play {
                            Some(p) => play_str(p),
                            // Terminal/unsearchable position (e.g. a state that is
                            // already won): search yields no action, and the
                            // protocol still requires a well-formed `bestmove`
                            // line — `(none)` is the graceful sentinel for that.
                            None => "(none)".to_string(),
                        };
                        emit(out, &format!("bestmove {}", mv));
                        let _ = handle.join();
                        break;
                    }
                    match lines.try_recv() {
                        Ok(l) => {
                            let l = trim_line(&l);
                            let cmd = l.split_whitespace().next().unwrap_or("");
                            match cmd {
                                "stop" => {
                                    if !stop_sent {
                                        stop_flag.store(true, Ordering::Relaxed);
                                        stop_sent = true;
                                    }
                                }
                                "isready" => {
                                    emit(out, "readyok");
                                }
                                "quit" => {
                                    stop_flag.store(true, Ordering::Relaxed);
                                    stop_sent = true;
                                    quit_requested = true;
                                }
                                // Unknown/other lines are ignored mid-search;
                                // the protocol is strict enough that only the
                                // three cases above are expected here.
                                _ => {}
                            }
                        }
                        Err(TryRecvError::Empty) => {}
                        Err(TryRecvError::Disconnected) => {
                            // stdin reader went away; just keep waiting for the
                            // search to finish and emit its result.
                        }
                    }
                    // Small bounded wait before re-polling; keeps this a pure
                    // std design (no `select`/`epoll` needed) while still
                    // responsive enough for the (human-scale) 1-request-per-turn
                    // protocol.
                    thread::sleep(std::time::Duration::from_millis(5));
                }
            }
            other => {
                emit(out, &format!("error unknown command: {}", other));
            }
        }
    }
}

/// Entry point for the `ai` binary: read lines on stdin, drive the protocol.
pub fn run() {
    let (tx, rx) = mpsc::channel::<String>();

    let reader = thread::Builder::new()
        .name("stdin-reader".into())
        .spawn(move || {
            let stdin = std::io::stdin();
            let mut reader = std::io::BufReader::new(stdin.lock());
            let mut line = String::new();
            loop {
                let n = match reader.read_line(&mut line) {
                    Ok(0) => break, // EOF
                    Ok(n) => n,
                    Err(_) => break,
                };
                let _ = n;
                let trimmed = trim_line(&line).to_string();
                line.clear();
                if tx.send(trimmed).is_err() {
                    break; // main thread gone
                }
            }
        })
        .expect("failed to spawn stdin reader thread");

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    main_loop(&mut out, rx);
    let _ = reader; // daemon-like reader; process exit reaps it.
}
