"""Three-way comparison: published / native / wrapped.

Reads success rates from the eval output directories and prints an
agreement table. CI math uses normal-approximation for proportions.
"""

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple


PUBLISHED_RATE = 0.220
PUBLISHED_CI = 0.026  # ±2.6% from MolmoSpaces leaderboard, MS-Pick


def _ci(rate: float, n: int, z: float = 1.96) -> float:
    """Normal-approximation 95% CI for a binomial proportion."""
    if n == 0:
        return float('inf')
    return z * math.sqrt(rate * (1 - rate) / n)


def _read_success(output_dir: Path) -> Optional[Tuple[int, int]]:
    """Look for `Success rate` lines in the run log; return (success, total)."""
    log = output_dir / 'run.log'
    if not log.exists():
        return None
    text = log.read_text()
    # Match e.g.: "Evaluation complete: 22/100"
    import re
    m = re.search(r'Evaluation complete:\s*(\d+)\s*/\s*(\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native_dir', default='~/molmo_validation/results/native')
    ap.add_argument('--wrapped_dir', default='~/molmo_validation/results/wrapped')
    args = ap.parse_args()

    native = _read_success(Path(args.native_dir).expanduser())
    wrapped = _read_success(Path(args.wrapped_dir).expanduser())

    print("=" * 72)
    print("MolmoSpaces MS-Pick — π0-FAST DROID — three-way comparison")
    print("=" * 72)
    print(f"{'Source':<24} {'Rate':>8} {'95% CI':>10} {'Episodes':>10}")
    print("-" * 72)
    print(f"{'Published (leaderboard)':<24} {PUBLISHED_RATE:>8.1%} "
          f"±{PUBLISHED_CI:>5.1%}     {'~975':>10}  (inferred from CI)")

    if native:
        s, n = native
        rate = s / n
        ci = _ci(rate, n)
        print(f"{'Native (stock cfg)':<24} {rate:>8.1%} ±{ci:>5.1%}     "
              f"{n:>10}")
    else:
        print(f"{'Native (stock cfg)':<24} {'(no run yet)':>26}")

    if wrapped:
        s, n = wrapped
        rate = s / n
        ci = _ci(rate, n)
        print(f"{'Wrapped (UMI schema)':<24} {rate:>8.1%} ±{ci:>5.1%}     "
              f"{n:>10}")
    else:
        print(f"{'Wrapped (UMI schema)':<24} {'(no run yet)':>26}")

    print("-" * 72)

    if native and wrapped:
        n_rate = native[0] / native[1]
        w_rate = wrapped[0] / wrapped[1]
        delta = abs(n_rate - w_rate)
        joint_ci = _ci(n_rate, native[1]) + _ci(w_rate, wrapped[1])
        print()
        print(f"native vs wrapped delta: {delta:>+.3f} "
              f"(joint 95% CI: ±{joint_ci:.3f})")
        if delta < joint_ci:
            print("   → schema is consistent with native (within CI)")
        else:
            print("   → schema diverges from native (>1 CI gap; investigate)")
        # Anchor against published
        n_anchor = abs(n_rate - PUBLISHED_RATE)
        n_anchor_ci = _ci(n_rate, native[1]) + PUBLISHED_CI
        print(f"native vs published delta: {n_anchor:>+.3f} "
              f"(joint 95% CI: ±{n_anchor_ci:.3f})")
        if n_anchor < n_anchor_ci:
            print("   → native run reproduces the leaderboard (within CI)")
        else:
            print("   → native run diverges from leaderboard; check setup")
    print("=" * 72)


if __name__ == '__main__':
    main()
