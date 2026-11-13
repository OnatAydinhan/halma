#!/usr/bin/env python
"""Stdlib analysis of the 480-cap double-blind (identical-weights) probe.

Both sides used the SAME default weight file, so any P1-vs-P2 split in
FINISHED games is pure first-mover / initiative bias (no weight difference).

Units:
  * finished games -> binomial test under H0 = fair 50/50
  * ongoing games  -> treated as 0.5/0.5 (the tune-loop 'ongoing = draw' convention)

Outputs: per-batch and pooled splits, exact two-sided binomial p,
and the equivalent Elo delta of the move-order advantage under both
'finished-only' and 'ongoing-as-draw' scorings.
"""
import math

def log10(x):
    return math.log(x) / math.log(10)

def binom_pmf(k, n, p=0.5):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def exact_two_sided_p(k, n, p=0.5):
    # two-sided exact: double the *rarer* tail (j <= min(k, n-k)); cap at 1.
    lo = min(k, n - k)
    tail = sum(binom_pmf(j, n, p) for j in range(0, lo + 1))
    return min(1.0, 2 * tail)

def elo_delta(S):
    return 400 * log10(S / (1 - S))

batches = [
    ("b1 seed=7000", 9, 3, 18),   # P1 wins, P2 wins, ongoing
    ("b2 seed=7001", 8, 4, 18),
]

print("first-mover (P1) bias — identical weights both sides, cap 480")
print("=" * 68)
p1f = 0
p2f = 0
ong_tot = 0
for name, p1w, p2w, ong in batches:
    fin = p1w + p2w
    print(f"\n{name}")
    print(f"  finished: P1={p1w}  P2={p2w}  (P1 share {p1w/fin:.1%})")
    print(f"  exact two-sided p vs fair 50/50 (n={fin}): {exact_two_sided_p(p1w, fin):.4f}")
    S_fin = p1w / fin
    S_drw = (p1w + 0.5 * ong) / (p1w + p2w + ong)   # tune-loop convention (ongoing = 0.5)
    print(f"  S finished-only = {S_fin:.3f}  ->  EloDelta = {elo_delta(S_fin):+.1f}")
    print(f"  S ongoing-as-draw = {S_drw:.3f}  ->  EloDelta = {elo_delta(S_drw):+.1f} (comparable to tune R)")
    p1f += p1w
    p2f += p2w
    ong_tot += ong

print("\n" + "=" * 68)
print("POOLED (both batches; the two orders are independent samples)")
fin = p1f + p2f
print(f"  games={p1f+p2f+ong_tot}  finished: P1={p1f}  P2={p2f}  (P1 share {p1f/fin:.1%})")
print(f"  ongoing={ong_tot}")
print(f"  exact two-sided p vs fair 50/50 (n={fin}): {exact_two_sided_p(p1f, fin):.4f}")
S_fin = p1f / fin
S_drw = (p1f + 0.5 * ong_tot) / (p1f + p2f + ong_tot)
print(f"  S finished-only   = {S_fin:.4f}  ->  move-order EloDelta = {elo_delta(S_fin):+.1f}")
print(f"  S ongoing-as-draw = {S_drw:.4f}  ->  move-order EloDelta = {elo_delta(S_drw):+.1f}  <-- headline")
