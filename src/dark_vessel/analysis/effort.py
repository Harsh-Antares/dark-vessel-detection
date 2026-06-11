"""Stage 5b — from snapshots to hours, carefully.

Each SAR pass is one instantaneous frame; Sentinel-1 revisits a given area
only every few days. Turning "k dark vessels seen in n snapshots" into
"estimated illegal-fishing HOURS over the window" is an extrapolation,
and we treat it as one — with explicit assumptions and uncertainty bounds.

The model
---------
Assume dark fishing vessels arrive in the MPA as a Poisson process and a
vessel present at a random instant is seen by the snapshot. Then the count
per snapshot k_i ~ Poisson(lambda), where lambda is the average number of
dark vessels present at any moment ("instantaneous abundance").

Expected dark-vessel-hours over an observation window of T hours:

    H = lambda * T        (vessels present, on average, x hours elapsed)

Uncertainty has two parts, both propagated:
 1. Sampling noise in lambda: with few snapshots, the Poisson mean is
    poorly determined. We bootstrap the snapshot counts (and fall back to
    the exact Gamma posterior for tiny n).
 2. Detection imperfection: divide by detector recall on the dark subset
    (measured in evaluation) to correct for missed vessels — a stated
    assumption, not a hidden fudge.

Optionally, an "active fishing hours" variant multiplies by the mean
fraction of present-time actually spent fishing (mean_visit_hours /
residence assumptions in the config) — keep both numbers separate and
label them clearly. Reviewers respect a careful estimate with error bars
far more than a confident single number. (Reference framing: Global
Fishing Watch's "apparent fishing effort".)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EffortEstimate:
    lam_mean: float          # mean dark vessels present per snapshot
    lam_lo: float
    lam_hi: float
    hours_mean: float        # estimated dark-vessel-hours over the window
    hours_lo: float
    hours_hi: float
    window_hours: float
    n_snapshots: int
    dark_recall_used: float
    assumptions: str

    def summary(self) -> str:
        return (
            f"Dark-vessel pressure estimate\n"
            f"  snapshots (SAR passes): {self.n_snapshots}\n"
            f"  mean dark vessels present per pass: "
            f"{self.lam_mean:.2f}  [{self.lam_lo:.2f}, {self.lam_hi:.2f}] (95% CI)\n"
            f"  observation window: {self.window_hours:.0f} h\n"
            f"  estimated dark-vessel-hours: "
            f"{self.hours_mean:,.0f}  [{self.hours_lo:,.0f}, {self.hours_hi:,.0f}]\n"
            f"  detector dark-recall correction: {self.dark_recall_used:.2f}\n"
            f"  ASSUMPTIONS: {self.assumptions}"
        )


def estimate_dark_hours(counts_per_pass: list[int] | np.ndarray,
                        window_hours: float,
                        dark_recall: float = 1.0,
                        bootstrap_iters: int = 2000,
                        seed: int = 0) -> EffortEstimate:
    """Estimate dark-vessel-hours inside one MPA with uncertainty bounds.

    counts_per_pass : dark detections inside the MPA for each SAR pass.
    window_hours    : the time span the estimate extrapolates over.
    dark_recall     : measured detector recall on dark vessels; counts are
                      divided by it to correct for missed targets.
    """
    counts = np.asarray(counts_per_pass, dtype=float)
    n = len(counts)
    if n == 0:
        raise ValueError("Need at least one SAR pass.")
    dark_recall = float(np.clip(dark_recall, 0.05, 1.0))
    corrected = counts / dark_recall

    rng = np.random.default_rng(seed)
    if n >= 3:
        # Bootstrap the per-pass counts.
        boot = rng.choice(corrected, size=(bootstrap_iters, n), replace=True)
        lam_samples = boot.mean(axis=1)
    else:
        # Too few passes to bootstrap: Gamma posterior for a Poisson rate
        # with Jeffreys prior, Gamma(total + 0.5, n).
        lam_samples = rng.gamma(corrected.sum() + 0.5, 1.0 / n,
                                size=bootstrap_iters)

    lam_mean = float(corrected.mean())
    lam_lo, lam_hi = (float(np.percentile(lam_samples, q)) for q in (2.5, 97.5))

    hours = lam_samples * window_hours
    return EffortEstimate(
        lam_mean=lam_mean,
        lam_lo=lam_lo,
        lam_hi=lam_hi,
        hours_mean=lam_mean * window_hours,
        hours_lo=float(np.percentile(hours, 2.5)),
        hours_hi=float(np.percentile(hours, 97.5)),
        window_hours=window_hours,
        n_snapshots=n,
        dark_recall_used=dark_recall,
        assumptions=(
            "(1) dark-vessel presence is stationary over the window; "
            "(2) each SAR pass is an unbiased instantaneous sample; "
            "(3) counts corrected by measured dark-recall; "
            "(4) 'hours' = vessel-presence-hours, not confirmed fishing time. "
            "This is an extrapolation, not a measurement."
        ),
    )
