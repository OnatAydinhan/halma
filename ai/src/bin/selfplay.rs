//! Self-play driver (T-9).
//!
//! Plays check_halma self vs self and dumps a **replayable, deterministic**
//! training CSV — one row per *move*, with the six named eval-term values at
//! the position before the move and the eventual game outcome from the mover's
//! point of view. Consumed later by `train_weights.py` (T-10).
//!
//! Deterministic by design: the T-9 gate requires that two runs with the same
//! `--seed` produce **byte-identical** CSVs. A wall-clock `movetime` budget makes
//! the *reached depth* depend on machine load, which would break that guarantee;
//! so this driver searches at a **fixed depth** (the reproducible axis) and only
//! enables a time budget when the caller explicitly opts in via `--movetime N`
//! with N > 0 (documented as non-deterministic across machines). The search
//! itself is a fresh, stateless `Search` per move — exactly how `server.rs`
//! drives it — so there is no hidden cross-move state.
//!
//! std-only. No protocol, no UI.

use halma8_ai::engine::{Board, Move};
use halma8_ai::evaluate::{base_presence, defense, mobility, progress, spread, threat, Weights};
use halma8_ai::search::{Play, Search};
use halma8_ai::types::{idx, row_col};

/// Exploration rate numerator/denominator. Each move: with probability
/// `EXPL / (EXPL + (100-EXPL))` the mover plays a uniformly random *legal*
/// action (diversity / off-policy data); otherwise it plays the search's best
/// move (the informative, on-policy rows). Both branches are driven by the same
/// seeded RNG, so the sequence of choices is fully deterministic for a `--seed`.
const EXPL_NUM: u64 = 35;
const EXPL_DEN: u64 = 100;

/// A single deterministic RNG (SplitMix64). Given the seed, the sequence of
/// `below` / `range` draws is reproducible byte-for-byte across runs.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        // Fold a per-run constant in so distinct seeds do not collide and a
        // zero seed still yields a well-mixed, non-degenerate stream.
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15).wrapping_mul(0xBF58476D1CE4E5B9),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, n)` (n >= 1). Low-bit quality is more than enough
    /// for action selection.
    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    /// `true` with probability `num/den` (den >= 1, num <= den).
    fn below(&mut self, num: u64, den: u64) -> bool {
        self.next_u64() % den < num
    }
}

/// Wire-format move string (identical shape to `server.rs::play_str` and the
/// plan's "Move string" section): `step r,c nr,nc` / `jump r,c mr,mc nr,nc` /
/// `stop`.
fn play_str(p: &Play) -> String {
    match p {
        Play::Step { from, to } => format!("step {},{} {},{}", from.0, from.1, to.0, to.1),
        Play::Jump { from, mid, to } => format!(
            "jump {},{} {},{} {},{}",
            from.0, from.1, mid.0, mid.1, to.0, to.1
        ),
        Play::Stop => "stop".to_string(),
    }
}

/// The engine-legal action set at `b`. During a chain only the chain piece
/// may jump (destination not in the revisit-ban) or `Stop`; otherwise the full
/// step+jump set. Matches `search::gen_actions_into` exactly.
fn gen_actions(b: &Board) -> Vec<Play> {
    let mut v = Vec::new();
    if b.chain_active {
        if let Some((cr, cc)) = b.chain_current {
            for &(mid, to) in b.legal_jumps(cr, cc).iter() {
                if !b.chain_banned[idx(to.0, to.1)] {
                    v.push(Play::Jump { from: (cr, cc), mid, to });
                }
            }
        }
        v.push(Play::Stop);
        return v;
    }
    for i in 0..64 {
        let pcell = b.grid[i];
        if pcell == 0 || pcell != b.turn {
            continue;
        }
        let (r, c) = row_col(i);
        for &to in b.legal_steps(r, c).iter() {
            v.push(Play::Step { from: (r, c), to });
        }
        for &(mid, to) in b.legal_jumps(r, c).iter() {
            v.push(Play::Jump { from: (r, c), mid, to });
        }
    }
    v
}

/// Apply a search action through the public engine authority.
fn apply(b: &mut Board, p: Play) {
    match p {
        Play::Stop => b.end_turn(),
        Play::Step { from, to } => {
            let m = Move::Step { from, to };
            let _ = b.apply_move(&m);
        }
        Play::Jump { from, mid, to } => {
            let m = Move::Jump { from, mid, to };
            let _ = b.apply_move(&m);
        }
    }
}

/// Minimal CSV escaping: quote a field only if it needs it (commas appear in
/// the position's chain token and in every `step`/`jump` move string).
fn esc(s: &str) -> String {
    if s
        .bytes()
        .any(|b| b == b',' || b == b'"' || b == b'\n' || b == b'\r')
    {
        let mut o = String::with_capacity(s.len() + 2);
        o.push('"');
        for ch in s.chars() {
            if ch == '"' {
                o.push('"');
            }
            o.push(ch);
        }
        o.push('"');
        o
    } else {
        s.to_string()
    }
}

struct Row {
    game_id: u32,
    ply: u32,
    mover: u8,
    pos: String,
    f: [i32; 6],
    mv: String,
    winner: Option<u8>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut games: u32 = 100;
    let mut depth: i32 = 6; // T-10.5: representative depth (was 4; d6 ≈ 20-50 ms/move on this box)
    let mut movetime: u64 = 0; // 0 => deterministic fixed-depth search (T-9 guarantee)
    let mut seed: u64 = 0;
    let mut out: Option<String> = None;
    let mut maxmove: u32 = 120; // per-game move cap (termination bound); extra flag
    // T-12: number of worker threads. 1 = today's serial behaviour. Games are
    // fanned out across these threads; determinism (byte-identical CSV for the
    // same seed+games) must hold for ANY K and is K-independent by construction.
    let mut workers: u32 = 1;
    // T-11 head-to-head: when BOTH `--weights-a` and `--weights-b` are given,
    // P1 plays with A's weight set and P2 with B's (each loaded once, then
    // applied to every move). Either one missing reverts to plain self-play
    // (both sides default) — head-to-head is all-or-nothing by design.
    let mut weights_a: Option<String> = None;
    let mut weights_b: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        let a = args[i].clone();
        match a.as_str() {
            "--games" => {
                if i + 1 < args.len() {
                    games = args[i + 1].parse().unwrap_or(games);
                }
                i += 2;
            }
            "--depth" => {
                if i + 1 < args.len() {
                    depth = args[i + 1].parse().unwrap_or(depth);
                }
                i += 2;
            }
            "--movetime" => {
                if i + 1 < args.len() {
                    movetime = args[i + 1].parse().unwrap_or(movetime);
                }
                i += 2;
            }
            "--seed" => {
                if i + 1 < args.len() {
                    seed = args[i + 1].parse().unwrap_or(seed);
                }
                i += 2;
            }
            "--maxmove" => {
                if i + 1 < args.len() {
                    maxmove = args[i + 1].parse().unwrap_or(maxmove);
                }
                i += 2;
            }
            "--workers" => {
                if i + 1 < args.len() {
                    workers = args[i + 1].parse().unwrap_or(workers);
                }
                i += 2;
            }
            s if s.starts_with("--workers=") => {
                workers = s.trim_start_matches("--workers=")
                    .parse()
                    .unwrap_or(workers);
                i += 1;
            }
            "--out" => {
                if i + 1 < args.len() {
                    out = Some(args[i + 1].clone());
                }
                i += 2;
            }
            "--weights-a" => {
                if i + 1 < args.len() {
                    weights_a = Some(args[i + 1].clone());
                }
                i += 2;
            }
            "--weights-b" => {
                if i + 1 < args.len() {
                    weights_b = Some(args[i + 1].clone());
                }
                i += 2;
            }
            s if s.starts_with("--out=") => {
                out = Some(s.trim_start_matches("--out=").to_string());
                i += 1;
            }
            s if s.starts_with("--weights-a=") => {
                weights_a = Some(s.trim_start_matches("--weights-a=").to_string());
                i += 1;
            }
            s if s.starts_with("--weights-b=") => {
                weights_b = Some(s.trim_start_matches("--weights-b=").to_string());
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    let out_file = out.unwrap_or_else(|| "selfplay.csv".to_string());

    if depth < 1 {
        depth = 1;
    }
    if maxmove < 1 {
        maxmove = 1;
    }

    // T-11: head-to-head needs BOTH side weight files; otherwise plain self-play.
    let h2h: Option<(Weights, Weights)> = match (&weights_a, &weights_b) {
        (Some(a), Some(b)) => Some((Weights::load(a), Weights::load(b))),
        _ => None,
    };

    let mut buf = String::new();
    buf.push_str(
        "game_id,ply,mover,position_string,progress,base_presence,defense,spread,mobility,threat,chosen_move,result_from_movers_view\n",
    );

    let mut total_rows = 0u64;
    let mut wins_p1 = 0u64;
    let mut wins_p2 = 0u64;
    let mut ongoing = 0u64;
    let mut move_durs_ms: Vec<u64> = Vec::new(); // per search-move wall clock (T-11 median-move-ms)

    // T-12: every game is played by a purely self-contained function whose RNG is
    // seeded from `splitmix_fold(seed, game_index)` — independent of any shared
    // cross-game stream and of thread scheduling. So the result of game `g` is
    // identical regardless of K. We fan the games out across `workers` threads
    // (each game written into its own locked slot), then emit rows in `game_index`
    // order → byte-identical CSV for the same `--seed`/`--games` at any K.
    let games_usize = games as usize;
    let slots: Vec<std::sync::Mutex<Option<GameOutcome>>> =
        (0..games_usize).map(|_| std::sync::Mutex::new(None)).collect();
    let workers = workers.max(1) as usize;

    if workers <= 1 {
        for g in 0..games_usize {
            let os = splitmix_fold(seed, g as u64);
            let co = play_game(g as u32, os, depth, movetime, maxmove, h2h);
            *slots[g].lock().unwrap() = Some(co);
        }
    } else {
        let next_idx = std::sync::atomic::AtomicUsize::new(0);
        let n_threads = workers.min(games_usize).max(1);
        let slots = &slots;
        std::thread::scope(|scope| {
            for _ in 0..n_threads {
                let next_idx = &next_idx;
                let slots = slots; // shared `&` to the slot Vec (Copy)
                let h2h = h2h; // fresh copy for this closure (Weights is Copy)
                scope.spawn(move || {
                    loop {
                        let g = next_idx.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if g >= games_usize {
                            break;
                        }
                        let os = splitmix_fold(seed, g as u64);
                        let co = play_game(g as u32, os, depth, movetime, maxmove, h2h);
                        *slots[g].lock().unwrap() = Some(co);
                    }
                });
            }
        });
    }

    // Emit in game_index order (independent of the order the threads finished).
    for g in 0..games_usize {
        let co = slots[g]
            .lock()
            .unwrap()
            .take()
            .expect("each game_index is claimed exactly once by atomic counter");
        for d in &co.move_durs_ms {
            move_durs_ms.push(*d);
        }
        for r in &co.rows {
            let label = match (r.winner, r.mover) {
                (Some(w), mover) if w == mover => "win",
                (Some(_), _) => "loss",
                (None, _) => "ongoing",
            };
            push_row(&mut buf, r, label);
            total_rows += 1;
        }
        match co.winner {
            Some(1) => wins_p1 += 1,
            Some(2) => wins_p2 += 1,
            // Some(other) is impossible for this game; count it as ongoing too.
            _ => ongoing += 1,
        }
    }

    if let Err(e) = std::fs::write(&out_file, &buf) {
        eprintln!("selfplay: failed to write {}: {}", out_file, e);
        std::process::exit(1);
    }

    println!(
        "selfplay: {} games, {} rows, P1 wins={}, P2 wins={}, ongoing={}",
        games, total_rows, wins_p1, wins_p2, ongoing
    );
    if let Some((wa, wb)) = &h2h {
        println!(
            "selfplay: head-to-head P1/A = [progress {} base_presence {} defense {} spread {} mobility {} threat {}]",
            wa.progress, wa.base_presence, wa.defense, wa.spread, wa.mobility, wa.threat
        );
        println!(
            "selfplay: head-to-head P2/B = [progress {} base_presence {} defense {} spread {} mobility {} threat {}]",
            wb.progress, wb.base_presence, wb.defense, wb.spread, wb.mobility, wb.threat
        );
    }
    if !move_durs_ms.is_empty() {
        let mut d = move_durs_ms;
        d.sort_unstable();
        println!("selfplay: median move ms: {}", d[d.len() / 2]);
    }
    if movetime > 0 {
        println!("selfplay: movetime={} enabled — output is NOT byte-reproducible across machines (use fixed --depth for reproducibility)", movetime);
    }
    println!("selfplay: wrote {}", out_file);
}

/// One game's full result. Self-contained so it can be produced on any thread.
struct GameOutcome {
    rows: Vec<Row>,
    winner: Option<u8>,
    move_durs_ms: Vec<u64>,
}

/// Deterministic per-game seed: folds the master seed with the game index so
/// each game gets its own independent, reproducible RNG stream (T-12). Because
/// this depends only on `(master, game_index)`, game `g` is identical at any
/// worker count and any thread schedule.
fn splitmix_fold(master: u64, index: u64) -> u64 {
    let mut z = master
        .wrapping_add(index)
        .wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Play one full game from the start position and return its rows, the final
/// winner, and the wall-clock durations of the search-moves it made. Purely
/// self-contained (no shared RNG / no global state), so it is safe to call from
/// many threads and its output does not depend on `K` or scheduling (T-12).
fn play_game(
    g: u32,
    game_seed: u64,
    depth: i32,
    movetime: u64,
    maxmove: u32,
    h2h: Option<(Weights, Weights)>,
) -> GameOutcome {
    let mut board = Board::start();
    let mut ply: u32 = 0;
    let mut rows: Vec<Row> = Vec::new();
    let mut winner: Option<u8> = None;
    let mut move_durs_ms: Vec<u64> = Vec::new();
    let mut rng = SplitMix64::new(game_seed);

    while ply < maxmove {
        let mover = board.turn;
        let actions = gen_actions(&board);
        if actions.is_empty() {
            // Defensive: a mover with no legal action (should be essentially
            // impossible for halma) ends the game as ongoing.
            break;
        }

        // Six named eval-term magnitudes for the mover, at the pre-move position.
        let f = [
            progress(&board, mover),
            base_presence(&board, mover),
            defense(&board, mover),
            spread(&board, mover),
            mobility(&board, mover),
            threat(&board, mover),
        ];
        let pos = board.to_position_str();

        // Choose the move: exploration (random legal) vs. search-best.
        let chosen = if rng.below(EXPL_NUM, EXPL_DEN) {
            actions[rng.range(actions.len())]
        } else {
            let mut s = Search::new();
            s.set_max_depth(depth);
            if movetime > 0 {
                s.set_movetime_ms(movetime);
            }
            // T-11 head-to-head: P1 (maximizer) uses A's weights, P2 (minimizer)
            // uses B's. `set_weights_pair` is a no-op at the default pair, so the
            // plain-self-play path is byte-identical.
            if let Some((wa, wb)) = &h2h {
                s.set_weights_pair(*wa, *wb);
            }
            let t_move = std::time::Instant::now();
            let (_score, play) = s.search(&board, depth);
            move_durs_ms.push(t_move.elapsed().as_millis() as u64);
            match play {
                Some(p) => p,
                None => actions[0],
            }
        };

        let mv = play_str(&chosen);
        apply(&mut board, chosen);

        rows.push(Row {
            game_id: g,
            ply,
            mover,
            pos,
            f,
            mv,
            winner: None,
        });
        ply += 1;

        if let Some(w) = board.winner {
            winner = Some(w);
            break;
        }
    }

    GameOutcome {
        rows,
        winner,
        move_durs_ms,
    }
}

fn push_row(buf: &mut String, r: &Row, label: &str) {
    buf.push_str(&r.game_id.to_string());
    buf.push(',');
    buf.push_str(&r.ply.to_string());
    buf.push(',');
    buf.push_str(&r.mover.to_string());
    buf.push(',');
    buf.push_str(&esc(&r.pos));
    for x in [r.f[0], r.f[1], r.f[2], r.f[3], r.f[4], r.f[5]] {
        buf.push(',');
        buf.push_str(&x.to_string());
    }
    buf.push(',');
    buf.push_str(&esc(&r.mv));
    buf.push(',');
    buf.push_str(label);
    buf.push('\n');
}
