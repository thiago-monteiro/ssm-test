from __future__ import annotations

import numpy as np
from scipy import stats


def paired_comparison(a, b, *, student_t: bool = False, reverse_test: bool = False):
    a, b = np.asarray(a), np.asarray(b)
    diff = b - a
    mean = float(diff.mean())
    sem = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    width = stats.t.ppf(0.975, len(diff) - 1) if student_t else 1.96
    t_stat, p_value = stats.ttest_rel(b, a) if reverse_test else stats.ttest_rel(a, b)
    return mean, [mean - width * sem, mean + width * sem], float(t_stat), float(p_value)


def paired_stats(vals_ordinary, vals_sphere):
    a = np.asarray([v for v in vals_ordinary if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in vals_sphere if np.isfinite(v)], dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    out = {
        "n": n,
        "mean_ordinary": float(a.mean()) if n else None,
        "mean_sphere": float(b.mean()) if n else None,
    }
    if n >= 2:
        mean, ci, t_stat, p_value = paired_comparison(a, b, reverse_test=True)
        out.update(
            mean_diff_sphere_minus_ordinary=mean,
            ci95=ci,
            t_stat=t_stat if (b - a).std() > 0 else None,
            p_value=p_value if (b - a).std() > 0 else None,
            n_seeds_positive_diff=int(((b - a) > 0).sum()),
        )
    return out


def compare_groups(frame, metrics, *, group="variant"):
    ordinary = frame[frame[group] == "ordinary"]
    sphere = frame[frame[group] == "sphere"]
    return {
        metric: paired_stats(ordinary[metric].tolist(), sphere[metric].tolist())
        for metric in metrics
        if metric in ordinary.columns
    }


def improvement(result, *, higher: bool):
    if not result or result.get("mean_diff_sphere_minus_ordinary") is None:
        return None
    diff = result["mean_diff_sphere_minus_ordinary"]
    better = diff > 0 if higher else diff < 0
    positives = result.get("n_seeds_positive_diff", 0)
    consistent = positives == (result.get("n", 0) if higher else 0)
    return dict(
        diff=diff,
        p=result.get("p_value"),
        sphere_better=bool(better),
        consistent_across_seeds=bool(consistent and better),
    )


def ci95_mean(values):
    if len(values) < 2:
        return np.nan, np.nan
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    width = sem * stats.t.ppf(0.975, len(values) - 1)
    return mean - width, mean + width
