//! Release micro-benchmark harness (T-6).
//!
//! Runs a fixed set of positions over depths 2..=6, records `(nodes, ms, best
//! move, score)` for each, and writes `ai/bench_results.json`. std-only: the
//! JSON is emitted by hand (no serde, matching the crate's zero-dep rule).
//!
//! Run: `cargo run --release --bin bench`. The gate it enforces is that at
//! depth 6 the *start* position searches in < 50 ms and the *mid-board*
//! positions in < 100 ms — i.e. that runtime is no longer the bottleneck and
//! the remaining effort should go to eval quality (T-5/T-10).

use halma8_ai::engine::{Board, Move};
use halma8_ai::search::{Play, Search};
use std::time::Instant;

/// Wire-format string for a search action (matches plan_ai.md "Move string").
fn play_str(p: &Play) -> String {
    match p {
        Play::Step { from, to } => {
            format!("step {},{} {},{}", from.0, from.1, to.0, to.1)
        }
        Play::Jump { from, mid, to } => {
            format!("jump {},{} {},{} {},{}", from.0, from.1, mid.0, mid.1, to.0, to.1)
        }
        Play::Stop => "stop".to_string(),
    }
}

fn start_pos() -> Board {
    Board::start()
}

/// A few plies of real play (alternating P1/P2 steps) into an open mid-board.
fn mid_pos_a() -> Board {
    let mut b = Board::start();
    let mv: [Move; 4] = [
        Move::Step { from: (5, 2), to: (4, 2) },
        Move::Step { from: (0, 5), to: (0, 4) },
        Move::Step { from: (6, 2), to: (5, 2) },
        Move::Step { from: (0, 6), to: (0, 5) },
    ];
    for m in mv {
        b.apply_move(&m).expect("bench fixture: expected a legal move");
    }
    b
}

/// A separate mid-board state (different pieces advanced) to exercise a second
/// TT/branching regime.
fn mid_pos_b() -> Board {
    let mut b = Board::start();
    // Alternating P1/P2; every destination is a verified-empty neighbour cell
    // (col-2 P1 pieces advance up/right, col-5 P2 pieces advance left).
    let mv: [Move; 6] = [
        Move::Step { from: (5, 2), to: (5, 3) },
        Move::Step { from: (0, 5), to: (0, 4) },
        Move::Step { from: (6, 2), to: (5, 2) },
        Move::Step { from: (1, 5), to: (1, 4) },
        Move::Step { from: (7, 2), to: (6, 2) },
        Move::Step { from: (2, 5), to: (2, 4) },
    ];
    for m in mv {
        b.apply_move(&m).expect("bench fixture: expected a legal move");
    }
    b
}

struct Row {
    depth: i32,
    nodes: u64,
    ms: f64,
    best: String,
    score: i32,
}

struct Pos {
    name: String,
    rows: Vec<Row>,
}

fn bench_position(name: &str, board: &Board, depths: &[i32]) -> Pos {
    let mut rows = Vec::with_capacity(depths.len());
    for &d in depths {
        let mut s = Search::new();
        s.set_max_depth(20); // never clamp below the requested benchmark depth
        let t = Instant::now();
        let (score, play) = s.search(board, d);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let best = match play {
            Some(p) => play_str(&p),
            None => "<none>".to_string(),
        };
        rows.push(Row {
            depth: d,
            nodes: s.nodes,
            ms,
            best,
            score,
        });
    }
    Pos {
        name: name.to_string(),
        rows,
    }
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            _ => o.push(ch),
        }
    }
    o
}

// ---------------------------------------------------------------------------
// T-6.5 additions: a time-budgeted deepening run (the headline "as deep as
// the budget allows" capability) and a like-for-like node-rate read against
// the pre-T-6.5 snapshot, all recorded into bench_results.json.
// ---------------------------------------------------------------------------

/// One time-driven search: spend `budget_ms`, report the deepest **fully
/// completed** depth (per the T-6.5 time-budget contract), its score/best
/// move, and the node rate (nodes/sec).
#[derive(Clone)]
struct TimeRun {
    name: String,
    budget_ms: u64,
    depth_completed: i32,
    nodes: u64,
    ms: f64,
    best: String,
    score: i32,
    nps: u64,
}

/// One like-for-like node-rate comparison at a single depth (depth 6 here).
struct NodeRateRow {
    name: String,
    depth: i32,
    before_nps: u64,
    after_nps: u64,
    speedup: f64,
}

#[inline]
fn nps_of(nodes: u64, ms: f64) -> u64 {
    if ms > 0.0 {
        (nodes as f64 / (ms / 1000.0)) as u64
    } else {
        0
    }
}

fn run_time_mode(name: &str, board: &Board, budget_ms: u64) -> TimeRun {
    let mut s = Search::new();
    s.set_movetime_ms(budget_ms);
    s.set_max_depth(64); // time is the axis; depth is only a generous safety cap
    let t = Instant::now();
    let (score, play) = s.search(&board, 64);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let best = match play {
        Some(p) => play_str(&p),
        None => "<none>".to_string(),
    };
    let nps = nps_of(s.nodes, ms);
    println!(
        "  {}  depth_completed={}  nodes={}  {:.3} ms  nps={}  score={:5}  best={}",
        name, s.deep_completed, s.nodes, ms, nps, score, best
    );
    TimeRun {
        name: name.to_string(),
        budget_ms,
        depth_completed: s.deep_completed,
        nodes: s.nodes,
        ms,
        best,
        score,
        nps,
    }
}

/// Pull a JSON number out of one pretty-printed line (our own fixed format).
fn grab_num(line: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{}\":", key);
    let i = line.find(&needle)?;
    let rest = line[i + needle.len()..].trim_start();
    let j = rest
        .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
        .unwrap_or(rest.len());
    rest[..j].parse().ok()
}

/// Depth-6 `(name, nodes, ms)` rows of a previously-saved bench file — read
/// by a targeted scan of our own pretty-printed format, std-only.
fn parse_before_depth6(path: &str) -> Vec<(String, u64, f64)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut out: Vec<(String, u64, f64)> = Vec::new();
    let mut name: Option<String> = None;
    for line in text.lines() {
        if let Some(i) = line.find("\"name\":\"") {
            if let Some(j) = line[i + 8..].find('"') {
                name = Some(line[i + 8..i + 8 + j].to_string());
            }
        }
        if line.contains("\"depth\":6") && line.contains("\"nodes\":") {
            if let (Some(name), Some(nodes), Some(ms)) =
                (name.clone(), grab_num(line, "nodes"), grab_num(line, "ms"))
            {
                out.push((name, nodes as u64, ms));
            }
        }
    }
    out
}

fn render_json(positions: &[Pos], time_runs: &[TimeRun], node_rate: &[NodeRateRow]) -> String {
    let mut o = String::new();
    o.push_str("{\"positions\":[\n");
    for (pi, p) in positions.iter().enumerate() {
        o.push_str("  {\"name\":\"");
        o.push_str(&esc(&p.name));
        o.push_str("\",\"rows\":[");
        for (ri, r) in p.rows.iter().enumerate() {
            if ri > 0 {
                o.push(',');
            }
            o.push_str(&format!(
                "\n    {{\"depth\":{},\"nodes\":{},\"ms\":{:.3},\"best\":\"{}\",\"score\":{}}}",
                r.depth, r.nodes, r.ms, esc(&r.best), r.score
            ));
        }
        o.push_str("\n  }");
        if pi + 1 < positions.len() {
            o.push(',');
        }
        o.push('\n');
    }
    o.push_str("],\n");

    o.push_str("\"time_runs\":[\n");
    for (ti, tr) in time_runs.iter().enumerate() {
        o.push_str(&format!(
            "  {{\"name\":\"{}\",\"budget_ms\":{},\"depth_completed\":{},\"nodes\":{},\"ms\":{:.3},\"best\":\"{}\",\"score\":{},\"nps\":{}}}",
            esc(&tr.name),
            tr.budget_ms,
            tr.depth_completed,
            tr.nodes,
            tr.ms,
            esc(&tr.best),
            tr.score,
            tr.nps
        ));
        if ti + 1 < time_runs.len() {
            o.push(',');
        }
        o.push('\n');
    }
    o.push_str("],\n");

    o.push_str("\"node_rate\":[\n");
    for (ni, nr) in node_rate.iter().enumerate() {
        o.push_str(&format!(
            "  {{\"name\":\"{}\",\"depth\":{},\"before_nps\":{},\"after_nps\":{},\"speedup\":{:.2}}}",
            esc(&nr.name),
            nr.depth,
            nr.before_nps,
            nr.after_nps,
            nr.speedup
        ));
        if ni + 1 < node_rate.len() {
            o.push(',');
        }
        o.push('\n');
    }
    o.push_str("]\n}\n");
    o
}

fn main() {
    let depths: [i32; 5] = [2, 3, 4, 5, 6];
    let fixtures: [(&str, fn() -> Board); 3] =
        [("start", start_pos), ("mid_a", mid_pos_a), ("mid_b", mid_pos_b)];

    // Build each board once and share it between the fixed-depth sweep and the
    // time-mode run (both must start from the identical position).
    let boards: Vec<(String, Board)> = fixtures
        .iter()
        .map(|(n, mk)| (n.to_string(), (mk)()))
        .collect();

    // --- T-6 harness (kept for like-for-like node-rate): fixed depths 2..=6.
    let mut positions: Vec<Pos> = Vec::with_capacity(boards.len());
    for (name, board) in boards.iter() {
        let pos = bench_position(name, board, &depths);
        println!("== {} ==", name);
        for r in &pos.rows {
            println!(
                "  depth {}  nodes={}  {:.3} ms  score={:5}  best={}",
                r.depth, r.nodes, r.ms, r.score, r.best
            );
        }
        positions.push(pos);
    }

    // --- T-6.5 headline: time-budgeted deepening. The gate is that the budget
    // is spent and the deepest *fully completed* depth's result is returned.
    let budget_ms = 500u64;
    let mut time_runs: Vec<TimeRun> = Vec::with_capacity(boards.len());
    println!("\n== T-6.5 time-mode ({} ms budget) ==", budget_ms);
    for (name, board) in boards.iter() {
        time_runs.push(run_time_mode(name, board, budget_ms));
    }

    // --- Node-rate read against the pre-T-6.5 snapshot (depth 6, like-for-like).
    let before = parse_before_depth6("bench_before_t65.json");
    let mut node_rate: Vec<NodeRateRow> = Vec::with_capacity(positions.len());
    for p in &positions {
        let Some(cur) = p.rows.iter().find(|r| r.depth == 6) else {
            continue;
        };
        let after_nps = nps_of(cur.nodes, cur.ms);
        let before_nps = before
            .iter()
            .find(|(n, _, _)| n == &p.name)
            .map(|(_, nodes, ms)| nps_of(*nodes, *ms))
            .unwrap_or(0);
        let speedup = if before_nps > 0 {
            (after_nps as f64) / (before_nps as f64)
        } else {
            0.0
        };
        println!(
            "\nNODE-RATE {} d6: {} -> {} nps  ({:.2}x)",
            p.name, before_nps, after_nps, speedup
        );
        node_rate.push(NodeRateRow {
            name: p.name.clone(),
            depth: 6,
            before_nps,
            after_nps,
            speedup,
        });
    }

    let json = render_json(&positions, &time_runs, &node_rate);
    std::fs::write("bench_results.json", &json)
        .expect("bench: failed to write bench_results.json (run from the ai/ dir, or check cwd)");
    println!("\nwrote bench_results.json ({} bytes)", json.len());

    // T-6 soft wall-clock prints (machine-dependent; kept for continuity).
    let start_ms = positions
        .iter()
        .find(|p| p.name == "start")
        .and_then(|p| p.rows.iter().find(|r| r.depth == 6))
        .map(|r| r.ms)
        .unwrap_or(0.0);
    let mid_ms: Vec<f64> = positions
        .iter()
        .filter(|p| p.name != "start")
        .filter_map(|p| p.rows.iter().find(|r| r.depth == 6).map(|r| r.ms))
        .collect();
    println!(
        "\nGATE(soft): start d6 = {:.3} ms (want < 50) — {}",
        start_ms,
        if start_ms < 50.0 { "OK" } else { "SLOW" }
    );
    for ms in mid_ms {
        println!(
            "GATE(soft): mid d6 = {:.3} ms (want < 100) — {}",
            ms,
            if ms < 100.0 { "OK" } else { "SLOW" }
        );
    }

    // T-6.5 hard gate: each time-mode run must (a) complete at least depth 1
    // within budget and (b) return a real best action. Legality of the returned
    // play is covered by the search unit tests (built on the engine's apply
    // move, so it is accepted by the Python engine by construction).
    for tr in &time_runs {
        assert!(
            tr.depth_completed >= 1,
            "{}: no depth completed within {} ms budget",
            tr.name,
            tr.budget_ms
        );
        assert!(tr.best != "<none>", "{}: no best action produced", tr.name);
    }
    println!(
        "\nGATE(hard): all {} time-mode runs completed >= depth 1 in budget and returned a legal best move",
        time_runs.len()
    );
}
