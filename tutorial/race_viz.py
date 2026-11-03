"""Accumulator-race animations for the tutorial, in the style of the project page.

Each animation has one row per trial: the stimulus on the left, the race of
the trained NRM on the right — the accumulators of one sampled trajectory, the
stimulus-dependent bound Θ(c), the model's crossing time t* and the human RT.
All rows share one clock, so an easy and a hard trial can be compared side by
side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from nrm.model import shifted_sigmoid

# Project-page palette
PAPER = "#F5F7FA"
PANEL = "#FFFFFF"
INK = "#1B2433"
MUTED = "#5B6576"
RULE = "#D5DBE3"
THR = "#C98A00"
ACC = ["#2B59C3", "#D64545", "#2A9D8F", "#8A5CC2"]
OTHER = "#C3CAD4"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": RULE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


# ── simulation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def simulate_race(model, covariates: dict, n_samples: int = 64, t: float = 0.993):
    """Sample ``n_samples`` races for ONE stimulus with the trained model.

    Uses ``simulate_nDDM`` exactly as the evaluators do (one SDE path per
    sample, soft crossing at probability ``t``). Returns the paths, the bound
    and, per sample, the crossing index and the choice.
    """
    model.eval()
    cov = {k: v.repeat_interleave(n_samples, dim=0) for k, v in covariates.items()}
    xs, ts, _, _ = model.simulate_nDDM(batch_size=n_samples, covariates=cov,
                                       soft=True, n_reps=1, t=t)
    theta = model.threshold_net(model.cond)              # (M,) or (M, T)
    theta = theta.reshape(n_samples, -1)
    margin = xs - theta.unsqueeze(-1) if theta.shape[1] > 1 else xs - theta.view(-1, 1, 1)
    crossed = (shifted_sigmoid(margin, model.kappa, 0.05) >= t).any(dim=2)   # (M, T)
    has = crossed.any(dim=1)
    idx = torch.where(has, crossed.float().argmax(dim=1),
                      torch.full_like(has, xs.shape[1] - 1, dtype=torch.long))
    choice = xs[torch.arange(n_samples), idx].argmax(dim=1)
    return {
        "xs": xs.cpu().numpy(),                       # (M, T, N)
        "ts": ts.cpu().numpy(),                       # (T,)
        "theta": theta.cpu().numpy(),                 # (M, 1) or (M, T)
        "cross_idx": idx.cpu().numpy(),               # (M,)
        "choice": choice.cpu().numpy(),               # (M,)
    }


def representative(sim: dict) -> int:
    """The sample with the modal choice whose RT is closest to the median RT."""
    rts = sim["ts"][sim["cross_idx"]]
    modal = np.bincount(sim["choice"]).argmax()
    cand = np.where(sim["choice"] == modal)[0]
    return int(cand[np.argmin(np.abs(rts[cand] - np.median(rts)))])


# ── animation ────────────────────────────────────────────────────────────────

@dataclass
class Row:
    """One trial: its stimulus and the race of one sampled trajectory."""
    sim: dict
    labels: list[str]
    colors: list[str]
    target: int
    title: str
    stimulus: object                  # (H, W[, 3]) image, or callable t -> image
    human_rt: float | None = None
    human_label: str = "human RT"
    deadline: float | None = None
    sample: int | None = None
    show_labels: list[int] | None = None   # accumulators to label (default: all)
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.sample is None:
            self.sample = representative(self.sim)


def _theta_path(row: Row) -> np.ndarray:
    th = row.sim["theta"][row.sample]
    return np.full(len(row.sim["ts"]), float(th[0])) if th.size == 1 else th


def animate(rows: list[Row], path, title: str = "", t_max: float | None = None,
            fps: int = 20, hold_s: float = 1.6, width: float = 9.6, dpi: int = 100):
    """Write a GIF with one stimulus + race row per trial, on a shared clock."""
    ts = rows[0].sim["ts"]
    t_max = t_max or float(ts[-1])
    t_end = max(max(float(ts[r.sim["cross_idx"][r.sample]]), r.human_rt or 0.0) for r in rows)
    t_end = min(t_max, t_end + 0.15)
    n_race = int(fps * max(2.5, 2.2 * t_end))           # slower than real time
    clock = np.concatenate([np.linspace(0, t_end, n_race),
                            np.full(int(fps * hold_s), t_end)])

    n = len(rows)
    fig = plt.figure(figsize=(width, 0.6 + 2.9 * n), dpi=dpi, facecolor=PAPER)
    if title:
        fig.text(0.035, 1 - 0.32 / (0.6 + 2.9 * n), title, fontsize=13,
                 color=INK, fontweight="bold", va="center")
    top = 1 - 0.6 / (0.6 + 2.9 * n)
    h = top / n
    arts = []
    for k, row in enumerate(rows):
        y0 = top - (k + 1) * h
        ax_s = fig.add_axes([0.035, y0 + 0.16 * h, 0.2, 0.72 * h])
        ax_r = fig.add_axes([0.31, y0 + 0.27 * h, 0.66, 0.6 * h])
        fig.text(0.035, y0 + 0.93 * h, row.title, fontsize=10.5, color=INK,
                 fontweight="bold", va="center")

        stim0 = row.stimulus(0.0) if callable(row.stimulus) else row.stimulus
        cmap = "gray" if np.ndim(stim0) == 2 else None
        im = ax_s.imshow(stim0, cmap=cmap, vmin=0, vmax=255, interpolation="bilinear")
        ax_s.set_xticks([]); ax_s.set_yticks([])
        for s in ax_s.spines.values():
            s.set_color(RULE)

        xs = row.sim["xs"][row.sample]                       # (T, N)
        th = _theta_path(row)
        ci = int(row.sim["cross_idx"][row.sample])
        win = int(row.sim["choice"][row.sample])
        lo = min(-0.2, float(xs[: ci + 1].min()) - 0.1)
        hi = max(float(th.max()), float(xs[: ci + 1].max())) * 1.15 + 0.05
        ax_r.set_xlim(0, t_max); ax_r.set_ylim(lo, hi)
        ax_r.set_facecolor(PANEL)
        for s in ("top", "right"):
            ax_r.spines[s].set_visible(False)
        ax_r.set_xlabel("time (s)"); ax_r.set_ylabel("evidence")
        ax_r.plot(ts, th, ls=(0, (5, 4)), color=THR, lw=1.8)
        ax_r.text(0.01 * t_max, th[0], " Θ(c)", color=THR, fontsize=9,
                  fontweight="bold", va="bottom")
        if row.deadline is not None:
            ax_r.axvline(row.deadline, color=MUTED, lw=1, ls=":")
            ax_r.text(row.deadline, lo + 0.02 * (hi - lo), f" deadline {row.deadline*1000:.0f} ms",
                      color=MUTED, fontsize=8, va="bottom")

        lines = []
        colors = list(row.colors)
        if colors[win] is None:              # a grey distractor won: show it in red
            colors[win] = ACC[1]
        for i in range(xs.shape[1]):
            c = colors[i] if colors[i] else OTHER
            lw = 1.4 if colors[i] else 0.8
            ln, = ax_r.plot([], [], color=c, lw=lw, alpha=1.0 if colors[i] else 0.8)
            lines.append(ln)
        hum = ax_r.axvline(row.human_rt if row.human_rt else 0, color=INK, lw=1.2,
                           ls=(0, (1, 2)), visible=False)
        hum_t = ax_r.text(row.human_rt or 0, hi, f" {row.human_label}", clip_on=False,
                          color=INK, fontsize=8, va="bottom", visible=False)
        dot, = ax_r.plot([], [], "o", color=colors[win], ms=7)
        tstar = ax_r.text(0, lo, "", color=INK, fontsize=9, fontweight="bold", va="top")
        labs = [ax_r.text(0, 0, "", fontsize=8.5, color=colors[i] or MUTED,
                          fontweight="bold" if i in (win, row.target) else None)
                for i in range(xs.shape[1])]
        out = fig.text(0.31, y0 + 0.03 * h, "", fontsize=9.5, color=MUTED)
        p_ok = float(np.mean(row.sim["choice"] == row.target))
        n_sim = len(row.sim["choice"])
        arts.append((row, im, lines, hum, hum_t, dot, tstar, labs, out, ci, win, xs, hi - lo,
                     p_ok, n_sim))

    def update(f):
        tc = clock[f]
        for (row, im, lines, hum, hum_t, dot, tstar, labs, out, ci, win, xs, span,
             p_ok, n_sim) in arts:
            j = min(int(np.searchsorted(ts, tc)), ci)
            if callable(row.stimulus):
                im.set_data(row.stimulus(min(tc, ts[ci])))
            for i, ln in enumerate(lines):
                ln.set_data(ts[: j + 1], xs[: j + 1, i])
                ln.set_alpha(0.45 if (j >= ci and i != win and row.colors[i]) else ln.get_alpha())
            show = (set(row.show_labels) | {win} if row.show_labels is not None
                    else range(xs.shape[1]))
            last = None
            for y, i in sorted(((xs[j, i], i) for i in show), reverse=True):
                yy = y if last is None else min(y, last - 0.08 * span)   # no overlaps
                labs[i].set_position((ts[j] + 0.02, yy)); labs[i].set_text(row.labels[i])
                last = yy
            if row.human_rt is not None and tc >= row.human_rt:
                hum.set_visible(True); hum_t.set_visible(True)
            if j >= ci:
                dot.set_data([ts[ci]], [xs[ci, win]])
                tstar.set_position((ts[ci] + 0.02, xs[ci, win] - 0.1 * span))
                tstar.set_text(f"t* = {ts[ci]*1000:.0f} ms")
                ok = "" if win == row.target else " (error)"
                out.set_text(f"Choice: {row.labels[win]}{ok} after {ts[ci]*1000:.0f} ms   ·   "
                             f"{p_ok:.0%} of {n_sim} simulated races choose "
                             f"{row.labels[row.target]}")
            else:
                out.set_text("Accumulating evidence…")
        return []

    anim = FuncAnimation(fig, update, frames=len(clock), interval=1000 / fps)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
