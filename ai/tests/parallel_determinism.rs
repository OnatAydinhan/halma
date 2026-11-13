//! T-12 gate: parallel self-play must be **worker-count independent**.
//!
//! The whole determinism argument rests on each game being seeded from
//! `splitmix_fold(master_seed, game_index)` and the rows being emitted in
//! `game_index` order. This test drives the real `selfplay` binary twice with
//! different `--workers` and asserts the CSVs are byte-identical — so the
//! guarantee does not silently regress into a scheduling-dependent one.

use std::path::Path;
use std::process::Command;

fn run_selfplay(out: &Path, workers: u32) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_selfplay"));
    cmd.args(["--games", "4", "--depth", "3", "--seed", "42", "--maxmove", "16"]);
    cmd.args(["--workers", &workers.to_string(), "--out", &out.to_string_lossy()]);
    let status = cmd.status().expect("selfplay binary should run");
    assert!(status.success(), "selfplay exited non-zero at workers={}", workers);
}

#[test]
fn output_is_worker_count_independent() {
    let dir = std::env::temp_dir().join(format!("halma_sp_wtest_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let a = dir.join("w2.csv");
    let b = dir.join("w5.csv");
    let c = dir.join("w1.csv");

    run_selfplay(&a, 2);
    run_selfplay(&b, 5);
    run_selfplay(&c, 1);

    let fa = std::fs::read(&a).expect("read workers=2 csv");
    let fb = std::fs::read(&b).expect("read workers=5 csv");
    let fc = std::fs::read(&c).expect("read workers=1 csv");

    assert!(!fa.is_empty(), "selfplay produced an empty CSV");
    assert_eq!(fa, fb, "workers=2 must be byte-identical to workers=5");
    assert_eq!(fa, fc, "parallel output must be byte-identical to the serial (workers=1) run");

    // best-effort cleanup
    let _ = std::fs::remove_dir_all(&dir);
}
