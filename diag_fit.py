#!/usr/bin/env python
"""T-13 -- throwaway READ-ONLY diagnostic.

Question: the self-play weight fit collapses to a single term ({defense:1}).
Is that because
  (A) int-round : the FLOAT fit has real multi-term shape, but
      train_weights.to_raw_weights does int(round(w_std/std)) and the large-
      scaled terms round to 0, flattening the learned vector;  OR
  (B) weak label: load_rows_progress's binary base-race label carries so little
      per-row signal that even the FLOAT coefficients are ~0 (only one term
      happens to clear 0.5), so a label change (T-14b signed target) is also
      required.

It reuses train_weights.py's exact functions (load_rows_progress, _standardize,
_fit, _split_indices, _predict, _accuracy, _f1, to_raw_weights) with the SAME
argument defaults tools_tune.py uses, so the fit here is bit-identical to what
the loop computed -- the reproduced int set should equal weights_v1.json.

Writes NOTHING, touches NO ai/ Rust source. ASCII-only output (cp1252 safe).

Usage:  py diag_fit.py [selfplay.csv]
"""
import csv
import sys

try:
    import train_weights as tw
except ImportError:  # pragma: no cover - fallback when CWD != project root
    sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__
                    else __file__.rsplit("/", 1)[0])
    import train_weights as tw


def count_games(path):
    games = set()
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r.get("game_id") is not None:
                games.add(int(r["game_id"]))
    return sorted(games)


def main(csv_path):
    mark, seed, holdout = 50, 0, 0.2       # train_weights.py defaults
    max_iter, lrm, lam = 5000, 0.5, 1e-3   # train_weights.py defaults

    games = count_games(csv_path)
    X, y, n_tied, n_bad = tw.load_rows_progress(csv_path, mark)
    n = len(X)
    n_cls1 = sum(1 for v in y if v == 1)
    n_cls0 = n - n_cls1

    print("=" * 78)
    print("T-13 diagnostic  (label=progress, mark=%d, seed=%d, holdout=%.2f)" %
          (mark, seed, holdout))
    print("=" * 78)
    print("csv           : %s" % csv_path)
    print("games         : %d   (game_id %s..%s)" % (len(games), games[0], games[-1]))
    print("labelled rows : %d   (tied dropped=%d, bad=%d)" % (n, n_tied, n_bad))
    print("class sizes   : mover-ahead(1)=%d   mover-behind(0)=%d" % (n_cls1, n_cls0))
    if n == 0:
        print("NO LABELLED ROWS -- nothing to fit.")
        return 0

    train_idx, test_idx = tw._split_indices(n, 1 - holdout, seed)
    mean, std = tw._standardize(X, train_idx)
    w_std, b = tw._fit(X, y, train_idx, mean, std, max_iter, lrm, lam, seed)
    float_w = [w_std[t] / std[t] for t in range(6)]
    int_w = [int(round(v)) for v in float_w]

    pred = tw._predict(w_std, b, X, mean, std)
    acc = tw._accuracy([y[i] for i in test_idx], [pred[i] for i in test_idx])
    f1 = tw._f1([y[i] for i in test_idx], [pred[i] for i in test_idx])

    print("")
    hdr = ("%15s %10s %10s %11s %12s %11s" %
           ("term", "mean(train)", "std(train)", "w_std(O1)", "float w", "int rnd"))
    print(hdr)
    print("-" * len(hdr))
    for t in range(6):
        print("%15s %10.3f %10.3f %11.4f %12.4f %11d" %
              (tw.TERMS[t], mean[t], std[t], w_std[t], float_w[t], int_w[t]))
    print("-" * len(hdr))

    print("")
    print("held-out accuracy: %.4f   (f1=%.4f, test n=%d)" % (acc, f1, len(test_idx)))
    print("reproduced int set: %s" % tw.to_raw_weights(w_std, std))

    # Bucket each (zeroed-by-round) term by how much float magnitude it actually had.
    survive = [tw.TERMS[t] for t in range(6) if int_w[t] != 0]
    trunc   = [tw.TERMS[t] for t in range(6) if int_w[t] == 0 and abs(float_w[t]) > 0.10]
    flat    = [tw.TERMS[t] for t in range(6) if int_w[t] == 0 and abs(float_w[t]) <= 0.10]
    n_round0 = sum(1 for v in int_w if v == 0)
    max_abs_float = max(abs(v) for v in float_w)
    max_abs_wstd = max(abs(v) for v in w_std)

    print("")
    print("round-to-0 count            = %d/6" % n_round0)
    print("survive (round to non-zero) : %d   %s" % (len(survive), survive))
    print("truncated (0<w<0.10..0.5)   : %d   %s" % (len(trunc), trunc))
    print("flat      (|float w|<=0.10) : %d   %s" % (len(flat), flat))
    print("max |float w| = %.4f     max |w_std| = %.4f" % (max_abs_float, max_abs_wstd))

    collapsed = all(v == 0 for v in int_w)
    at_chance = acc <= 0.5
    fit_kept = not collapsed and not at_chance
    print("collapsed(all int=0)=%s   at_chance(acc<=0.5)=%s   fit kept(no fallback)=%s" %
          (collapsed, at_chance, fit_kept))
    print("=" * 78)

    if max_abs_float < 0.5:
        verdict = ("(B) weak label -- even the FLOAT coefficients are all < 0.5 (max=%.4f); "
                   "per-row features barely track the base-race label. T-14 float alone is NOT "
                   "sufficient: also switch to the signed (progress_1 - progress_2) target (T-14b)."
                   % max_abs_float)
    elif len(trunc) >= 1 and len(flat) <= 3:
        verdict = ("(A) int-round -- the float fit has real shape (max|w|=%.4f; %d term(s) with "
                   "0.10<|w|<0.5 got zeroed: %s) but int(round) flattens it to %s. T-14 "
                   "(real-valued Weights) directly fixes this; T-14b not required."
                   % (max_abs_float, len(trunc), trunc, survive))
    else:
        verdict = ("(B) weak label -- only %d term(s) have any float magnitude (%s); the other "
                   "%d (incl. %s) are ~zero, i.e. a single-feature / near-noise fit. Treat as a "
                   "label-signal problem: T-14 float AND T-14b signed target."
                   % (len(survive), survive, len(flat), flat))
    print("VERDICT: " + verdict)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", default="selfplay.csv")
    a = ap.parse_args()
    sys.exit(main(a.csv))
