"""Build the tutorial animations from the trained NRMs.

For each benchmark one easy and one hard held-out trial are drawn from the
test split; the released NRM simulates its race for that very stimulus.

    python tutorial/make_gifs.py            # writes assets/nrm_{rdm,sc,sat}.gif

Needs the RDM/SC/SAT data and embeddings (``python fetch_assets.py``)
and, for the RDM video, opencv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nrm.config import build_model                     # noqa: E402
from tutorial.race_viz import ACC, Row, animate, simulate_race  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ASSETS = ROOT / "assets"
N_SAMPLES = 64
RDM_DOTS = 60          # illustrative video density (the experiment used 6)
SEED = 42


def _seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def load_nrm(benchmark: str, weight: str):
    model = build_model(benchmark, device=DEVICE)
    ckpt = torch.load(ROOT / "nrm" / "weights" / weight, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return model.eval()


def typical_trial(df, mask, rng):
    """A correct human trial with a typical RT for its condition."""
    sub = df[mask & (df["_correct"] == 1)]
    med = sub["_rt"].median()
    sub = sub[(sub["_rt"] - med).abs() <= 0.1 * med]
    return int(rng.choice(sub.index.values))


# ── RDM ──────────────────────────────────────────────────────────────────────

def rdm_rows():
    from datasets.rdm.loader import load_rdm_nrm_dataset
    from datasets.rdm.rdm_generator import RDM_generator

    ds = load_rdm_nrm_dataset(pkl_out=ROOT / "datasets/rdm/rdm_nrm_dataset.pkl",
                              embeddings_path=ROOT / "embeddings/rdm/rdm_embeddings.pt")
    ds.remove_rt_outliers_mad(mad_thresh=3.0, min_rt=0.05, max_rt=3.0)
    _, _, test = ds.train_val_test_split(val_size=0.10, test_size=0.10, seed=42)
    df = test.df.reset_index(drop=True)
    df["_correct"], df["_rt"] = df["correct"], df["rt"]
    model = load_nrm("rdm", "NRM_rdm.pth")
    rng = np.random.default_rng(SEED)
    labels = ["right", "left"]                 # direction 0° / 180°

    rows = []
    for coh, name in ((51.2, "Easy"), (3.2, "Hard")):
        i = typical_trial(df, np.isclose(df["coherence"], coh), rng)
        r = df.loc[i]
        # The experiment shows 6 dots per frame, too sparse to read at GIF frame
        # rates; the panel shows the trial's direction and coherence with
        # RDM_DOTS dots instead. The race is the model's response to the real
        # trial.
        np.random.seed(int(r["video_seed"]))
        video = RDM_generator(direction=int(r["direction"]), coherence=float(r["coherence"]),
                              nDot=RDM_DOTS)
        fps = model.fs                                     # 75 Hz
        stim = lambda t, v=video: (255 - v[min(int(t * fps), len(v) - 1)]).clip(0, 255)
        _seed()
        sim = simulate_race(model, {"img_embedding": test.embeddings[i:i + 1].to(DEVICE).float()},
                            n_samples=N_SAMPLES)
        rows.append(Row(sim=sim, labels=labels, colors=ACC[:2],
                        target=int(r["direction_label"]),
                        title=f"{name}: {coh:g}% coherence, dots move {labels[int(r['direction_label'])]}",
                        stimulus=stim, human_rt=float(r["rt"])))
    return rows


# ── SC ───────────────────────────────────────────────────────────────────────

def sc_rows():
    from datasets.sc.loader import SCDataset

    ds = SCDataset(csv_path=str(ROOT / "datasets/sc/experiment_1.csv"),
                   image_dir=str(ROOT / "datasets/sc/stimuli"), load_images=False)
    _, _, test = ds.train_val_test_split(val_size=0.10, test_size=0.20, seed=42)
    test.embeddings = torch.load(ROOT / "embeddings/sc/sc_embeddings.pt", weights_only=False)["test"]
    df = test.df.reset_index(drop=True)
    df["_correct"], df["_rt"] = (df["human_choice"] == df["choice"]).astype(int), df["rt"]
    model = load_nrm("sc", "NRM_sc.pt")
    rng = np.random.default_rng(SEED)
    labels = ["natural", "man-made"]

    rows = []
    for (cat, name) in (("natural_100", "Easy"), ("natural_20", "Hard")):
        i = typical_trial(df, df["cat"] == cat, rng)
        r = df.loc[i]
        img = np.asarray(Image.open(ROOT / "datasets/sc/stimuli" / r["img"]).convert("L"))
        _seed()
        sim = simulate_race(model, {"img_embedding": test.embeddings[i:i + 1].to(DEVICE).float()},
                            n_samples=N_SAMPLES)
        rows.append(Row(sim=sim, labels=labels, colors=ACC[:2], target=int(r["choice"]),
                        title=f"{name}: scene at discriminability {r['prob']:g} ({cat.replace('_', ' ')}%)",
                        stimulus=img, human_rt=float(r["rt"])))
    return rows


# ── SAT ──────────────────────────────────────────────────────────────────────

# CC0 photos from Wikimedia Commons (credited in README), shown in place of the
# ImageNet stimulus. They receive the same perturbation at the same level as the trial.
SAT_PHOTOS = {c: ROOT / "assets/stimuli" / f"{c}.jpg"
              for c in ("cat", "dog", "car", "boat")}


def sat_rows(task="color-blur", category="boat", level="3", filename="113_blur_3_boat.JPEG"):
    from datasets.sat.generate_images import _blur_tf
    from datasets.sat.loader import CATEGORIES, SATDataset

    ds = SATDataset(file_path=str(ROOT / f"datasets/sat/{task}.txt"),
                    image_dir=str(ROOT / "datasets/sat/data"), load_images=False)
    emb = torch.load(ROOT / f"embeddings/sat/sat_{task.replace('-', '_')}_embeddings.pt",
                     weights_only=False)
    key = {"/".join(Path(p).parts[-2:]): k for k, p in enumerate(emb["image_paths"])}
    df = ds.df
    df["_k"] = df["image_path"].map(lambda p: key.get("/".join(Path(p).parts[-2:])))
    model = load_nrm("sat", "NRM_sat_blur.pth")

    # One image seen under both the shortest and the longest deadline.
    cand = df[(df["category"] == category) & (df["level_str"] == level) & df["_k"].notna()]
    short, long_ = cand[cand["time_limit"] == 500], cand[cand["time_limit"] == 1500]
    common = sorted(set(short["filename"]) & set(long_["filename"]))
    fname = filename if filename in common else common[0]
    photo = _blur_tf(float(level))(Image.open(SAT_PHOTOS[category]).convert("RGB"))
    photo = np.asarray(photo)

    target = CATEGORIES.index(category)
    colors = [None] * len(CATEGORIES)          # distractors drawn in grey
    colors[target] = ACC[0]
    rows = []
    for tl, name in ((500, "Short deadline"), (1500, "Long deadline")):
        trials = df[(df["filename"] == fname) & (df["time_limit"] == tl)]
        r = trials.iloc[0]
        cov = {"img_embedding": emb["embeddings"][int(r["_k"])].view(1, -1).to(DEVICE).float(),
               "time_limit": torch.tensor([[tl / 1000.0]], device=DEVICE)}
        _seed()
        sim = simulate_race(model, cov, n_samples=N_SAMPLES)
        rows.append(Row(sim=sim, labels=list(CATEGORIES), colors=list(colors), target=target,
                        title=f"{name}: {tl} ms — {category}, blur σ = {level}",
                        stimulus=photo, human_rt=float(trials["duration"].median()) / 1000.0,
                        human_label=f"human RT (median of {len(trials)})",
                        deadline=tl / 1000.0, show_labels=[target]))
    return rows


def main():
    ASSETS.mkdir(exist_ok=True)
    animate(rdm_rows(), ASSETS / "nrm_rdm.gif", title="Random-dot motion", t_max=2.0)
    animate(sc_rows(), ASSETS / "nrm_sc.gif", title="Scene categorisation", t_max=1.5)
    animate(sat_rows(), ASSETS / "nrm_sat.gif", title="Speed–accuracy trade-off", t_max=2.0)
    for p in sorted(ASSETS.glob("nrm_*.gif")):
        print(f"{p.relative_to(ROOT)}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
