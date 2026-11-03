"""RDM dataset: loader, outlier removal and train/val/test split (§3)."""

import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import display_path

# ── RDM video generator ──────────────────────────────────────────────────────
from datasets.rdm.rdm_generator import RDM_generator


# ── Constants ─────────────────────────────────────────────────────────────────

# Median-centred coherence → original % (7 levels)
_COHERENCE_CENTERED = [-0.056, -0.048, -0.032, 0.000, 0.064, 0.192, 0.448]
COHERENCE_LIST      = [0.8,    1.6,    3.2,    6.4,   12.8,  25.6,  51.2]   # in %
COHERENCE_MAP       = {c: p for c, p in zip(_COHERENCE_CENTERED, COHERENCE_LIST)}

# RDM directions: 0° = rightward (label 0), 180° = leftward (label 1)
DIRECTIONS   = [0, 180]
DIR_TO_LABEL = {0: 0, 180: 1}

# Paths (relative to repository root)
_REPO_ROOT   = Path(__file__).parent.parent.parent
CSV_PATH     = Path(__file__).parent / "teddm_data.csv"
VIDEO_DIR    = _REPO_ROOT / "datasets" / "rdm" / "videos"
PKL_OUT      = Path(__file__).parent / "rdm_nrm_dataset.pkl"
ME_EMBS_OUT  = _REPO_ROOT / "embeddings" / "rdm" / "rdm_embeddings.pt"


# ── Dataset ───────────────────────────────────────────────────────────────────

class RDMDataset(Dataset):
    """
    PyTorch Dataset for the RDM task.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame produced by ``build_rdm_nrm_dataset``.
        Required columns: video_index, subID, coherence_idx, coherence,
        coherence_str, direction, direction_label, rt, correct, choice,
        human_choice, split, video_seed, isVGP.
    embeddings : torch.Tensor | None
        Pre-computed motion-energy embeddings, shape (N, T, D), row-aligned
        with df.
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        embeddings: torch.Tensor | None = None,
    ):
        self.df         = df.reset_index(drop=True)
        self.embeddings = embeddings

    # ------------------------------------------------------------------
    # PyTorch Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        sample = {
            "video_index":  torch.tensor(int(row["video_index"]),      dtype=torch.long),
            "choice":       torch.tensor(int(row["direction_label"]),  dtype=torch.long),
            "human_choice": torch.tensor(int(row["human_choice"]),     dtype=torch.long),
            "rt":           torch.tensor(float(row["rt"]),             dtype=torch.float32),
            "coherence":    torch.tensor(float(row["coherence"]),      dtype=torch.float32),
            "level_str":    str(row["coherence_str"]),
            "sub_id":       str(row["subID"]),
        }

        if self.embeddings is not None:
            sample["embedding"] = self.embeddings[idx]
            # shape: (T, D) float32  (motion energy)

        return sample

    # ------------------------------------------------------------------
    # Internal copy helper
    # ------------------------------------------------------------------

    def _shallow_copy(self):
        new_ds = RDMDataset.__new__(RDMDataset)
        for k, v in self.__dict__.items():
            setattr(new_ds, k, v)
        return new_ds

    # ------------------------------------------------------------------
    # Outlier removal
    # ------------------------------------------------------------------

    def remove_rt_outliers_mad(
        self,
        mad_thresh: float = 3.0,
        min_rt: float | None = 0.05,
        max_rt: float | None = 3.0,
    ):
        """
        MAD-based outlier removal on log(RT), applied per coherence level.

        Hard bounds (min_rt, max_rt) are applied first; then log(RT) outliers
        beyond mad_thresh * MAD are discarded.
        """
        before = len(self.df)
        keep   = np.zeros(len(self.df), dtype=bool)

        for coh_idx in self.df["coherence_idx"].unique():
            coh_mask    = (self.df["coherence_idx"] == coh_idx).values
            coh_indices = np.where(coh_mask)[0]
            rt          = self.df["rt"].values[coh_indices].astype(float)

            valid = np.ones(len(rt), dtype=bool)
            if min_rt is not None:
                valid &= rt >= min_rt
            if max_rt is not None:
                valid &= rt <= max_rt

            rt_v = rt[valid]
            if len(rt_v) < 2:
                keep[coh_indices[valid]] = True
                continue

            log_rt  = np.log(rt_v)
            median  = np.median(log_rt)
            mad_val = np.median(np.abs(log_rt - median))

            if mad_val == 0:
                keep[coh_indices[valid]] = True
                continue

            z  = np.abs(log_rt - median) / mad_val
            ok = z < mad_thresh
            keep[coh_indices[np.where(valid)[0][ok]]] = True

        # In-memory embeddings must be filtered to match the surviving rows.
        if self.embeddings is not None:
            self.embeddings = self.embeddings[torch.from_numpy(keep)]

        self.df  = self.df[keep].reset_index(drop=True)
        after    = len(self.df)
        print(
            f"RT outlier removal (MAD per coherence, thresh={mad_thresh}, "
            f"min_rt={min_rt}s, max_rt={max_rt}s): "
            f"{before - after} removed, {after} remaining."
        )

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------

    def train_val_test_split(
        self,
        val_size:  float = 0.10,
        test_size: float = 0.10,
        seed:      int   = 42,
    ):
        """
        Returns (train_ds, val_ds, test_ds).

        Split is stratified by (subID × coherence_idx) to ensure every
        subject and coherence level is represented in each split.

        The split column is pre-assigned in ``build_rdm_nrm_dataset``;
        this method just slices on it. If the 'split' column is absent
        (e.g. after a fresh build without saving) a random split is made here.

        **The stored column wins**, so ``val_size`` / ``test_size`` / ``seed``
        are inert whenever the pkl carries one; a mismatch between them and the
        stored split is reported. Rebuild the pkl to change the split — the
        arguments here cannot.

        Parameters
        ----------
        val_size  : fraction of data for validation (default 0.10)
        test_size : fraction of data for test (default 0.10)
        seed      : random seed for reproducibility
        """
        if "split" in self.df.columns:
            masks = {s: (self.df["split"] == s).values for s in ("train", "val", "test")}
            stored = {s: int(m.sum()) for s, m in masks.items()}
            n = len(self.df)
            asked = {"val": val_size, "test": test_size}
            for name, frac in asked.items():
                # _stratified_split floors per (subID × coherence) group, so the
                # realised fraction sits a little below the requested one; 2 pp
                # of slack keeps the check from firing on that rounding alone.
                if abs(stored[name] / n - frac) > 0.02:
                    warnings.warn(
                        f"RDMDataset.train_val_test_split: the pkl's stored "
                        f"'split' column gives {name}={stored[name]} "
                        f"({stored[name] / n:.1%} of {n}) but you asked for "
                        f"{frac:.0%}. The stored column wins and your argument "
                        f"is ignored. Rebuild with build_rdm_nrm_dataset("
                        f"val_size=..., test_size=..., seed=...) to change it.",
                        RuntimeWarning, stacklevel=2,
                    )
        else:
            masks = _stratified_split(self.df, val_size=val_size,
                                      test_size=test_size, seed=seed)

        result = {}
        for name, mask in masks.items():
            idxs   = np.where(mask)[0]
            ds     = self._shallow_copy()
            ds.df  = self.df.iloc[idxs].reset_index(drop=True)
            if self.embeddings is not None:
                ds.embeddings = self.embeddings[idxs]
            result[name] = ds
            print(f"  {name:5s}: {len(idxs):>6d} trials")

        return result["train"], result["val"], result["test"]

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    def generate_and_save_videos(
        self,
        video_dir: Path | str = VIDEO_DIR,
        overwrite: bool = False,
    ):
        """
        Pre-generates all RDM videos and saves them to *video_dir* as uint8
        NumPy arrays (.npy), shape (150, 126, 126).

        Filename: ``{split}_coh{coherence_idx:02d}_{video_index:07d}.npy``
        The random seed is read from the 'video_seed' column.
        """
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {len(self.df)} RDM videos → {display_path(video_dir)} ...")
        for idx in tqdm(range(len(self.df)), desc="Videos"):
            row  = self.df.iloc[idx]
            fname = (
                f"{row['split']}_"
                f"coh{int(row['coherence_idx']):02d}_"
                f"{int(row['video_index']):07d}.npy"
            )
            save_path = video_dir / fname
            if save_path.exists() and not overwrite:
                continue

            np.random.seed(int(row["video_seed"]))
            video = RDM_generator(
                direction=int(row["direction"]),
                coherence=float(row["coherence"]),
            ).astype(np.uint8)
            np.save(save_path, video)

        print(f"Done. Videos saved to {display_path(video_dir)}")

    @staticmethod
    def load_video(
        video_index:   int,
        split:         str,
        coherence_idx: int,
        video_dir:     Path | str = VIDEO_DIR,
    ) -> np.ndarray:
        """Loads a single pre-generated video (150, 126, 126) uint8."""
        video_dir = Path(video_dir)
        fname     = f"{split}_coh{coherence_idx:02d}_{video_index:07d}.npy"
        return np.load(video_dir / fname)

    # ------------------------------------------------------------------
    # Embedding computation (motion energy)
    # ------------------------------------------------------------------

    def compute_motion_energy_embeddings(
        self,
        save_path:    str | Path        = ME_EMBS_OUT,
        video_dir:    Path | str        = VIDEO_DIR,
        device:       str | None        = None,
        n_directions: int               = 8,
        speeds:       list | None       = None,
        lam:          float             = 8.0,
        sigma_xy:     float             = 4.0,
        sigma_t:      float             = 3.0,
        kx:           int               = 13,
        ky:           int               = 13,
        kt:           int               = 7,
    ):
        """
        Computes per-video motion energy embeddings (Adelson & Bergen 1985)
        and saves them to *save_path*.

        Each video yields a (T, D) tensor where T=150 (frames) and
        D = n_directions * len(speeds).

        Videos are loaded from *video_dir* if the corresponding .npy file
        exists; otherwise they are regenerated on-the-fly from the stored
        seed / direction / coherence columns (deterministic).

        Saved file structure:
            data = torch.load(save_path, weights_only=False)
            embeddings    = data["embeddings"]     # (N, T, D)
            video_indices = data["video_indices"]  # (N,) alignment key
            filter_params = data["filter_params"]  # dict — for bookkeeping
        """
        from datasets.rdm.motion_energy import MotionEnergyEmbedder, SPEEDS as DEFAULT_SPEEDS

        save_path = Path(save_path)
        video_dir = Path(video_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if speeds is None:
            speeds = DEFAULT_SPEEDS

        filter_kw = dict(lam=lam, sigma_xy=sigma_xy, sigma_t=sigma_t, kx=kx, ky=ky, kt=kt)

        print(f"Initialising MotionEnergyEmbedder on {device} ...")
        print(f"  n_directions={n_directions}  speeds={speeds}")
        print(f"  lam={lam}  sigma_xy={sigma_xy}  sigma_t={sigma_t}  "
              f"kernel=({kx},{ky},{kt})")
        embedder = MotionEnergyEmbedder(
            n_directions=n_directions,
            speeds=speeds,
            device=device,
            **filter_kw,
        )

        all_embs    = []
        all_indices = []

        print(f"Computing motion energy embeddings for {len(self.df)} videos ...")
        for idx in tqdm(range(len(self.df)), desc="Motion energy"):
            row       = self.df.iloc[idx]
            vid_index = int(row["video_index"])

            # Try to load pre-generated video from disk
            fname = f"{row['split']}_coh{int(row['coherence_idx']):02d}_{vid_index:07d}.npy"
            fpath = video_dir / fname
            if fpath.exists():
                video = np.load(str(fpath))
            else:
                # Deterministic on-the-fly fallback (same seed used at generation)
                np.random.seed(int(row["video_seed"]))
                video = RDM_generator(
                    direction=int(row["direction"]),
                    coherence=float(row["coherence"]),
                ).astype(np.uint8)

            emb = embedder.embed_video(video)   # (T, D) on CPU
            all_embs.append(emb)
            all_indices.append(vid_index)

        embeddings    = torch.stack(all_embs)                        # (N, T, D)
        video_indices = torch.tensor(all_indices, dtype=torch.long)  # (N,)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings":    embeddings,
                "video_indices": video_indices,
                "filter_params": {
                    "n_directions": n_directions,
                    "speeds": speeds,
                    **filter_kw,
                },
            },
            str(save_path),
        )
        print(f"Embeddings saved → {display_path(save_path)}  shape={tuple(embeddings.shape)}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _stratified_split(
    df: pd.DataFrame,
    val_size:  float = 0.10,
    test_size: float = 0.10,
    seed:      int   = 42,
) -> dict[str, np.ndarray]:
    """
    Stratified trial-level split by (subID × coherence_idx).

    Returns a dict with keys 'train', 'val', 'test' mapping to bool arrays
    of length len(df).
    """
    rng   = np.random.default_rng(seed)
    split = np.full(len(df), "train", dtype=object)

    for (sub, coh), grp in df.groupby(["subID", "coherence_idx"]):
        # pandas >= 2 returns a read-only view here, which `rng.shuffle`
        # cannot shuffle in place; the copy consumes the RNG identically.
        idxs = grp.index.to_numpy(copy=True)
        rng.shuffle(idxs)
        n    = len(idxs)
        n_te = max(1, int(n * test_size))
        n_va = max(1, int(n * val_size))
        split[idxs[:n_te]]           = "test"
        split[idxs[n_te:n_te + n_va]] = "val"
        # rest stays "train"

    return {
        "train": split == "train",
        "val":   split == "val",
        "test":  split == "test",
    }


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_rdm_nrm_dataset(
    csv_path:       Path | str = CSV_PATH,
    coherence_list: list[float] = COHERENCE_LIST,
    val_size:       float       = 0.10,
    test_size:      float       = 0.10,
    seed:           int         = 42,
    pkl_out:        Path | str | None = PKL_OUT,
) -> "RDMDataset":
    """
    Builds a metadata DataFrame from *teddm_data.csv* and wraps it in
    an ``RDMDataset``.

    Steps
    -----
    1. Load CSV and decode median-centred coherence → original % values.
    2. Assign a deterministic random direction (0°/180°) per trial.
    3. Derive human_choice from (direction_label, correct).
    4. Assign stratified train/val/test split labels.
    5. Optionally save to *pkl_out* for fast reloading.

    Parameters
    ----------
    csv_path       : path to teddm_data.csv
    coherence_list : original coherence % values in the same order as
                     COHERENCE_CENTERED (default: COHERENCE_LIST)
    val_size       : fraction of data for validation (default 0.10)
    test_size      : fraction of data for test (default 0.10)
    seed           : base random seed for direction assignment and split
    pkl_out        : if given, saves metadata DataFrame as pkl for reuse
    """
    csv_path = Path(csv_path)
    print(f"Loading human RT data from {display_path(csv_path)} ...")
    df_raw = pd.read_csv(csv_path)

    # Map centred coherence → (coh_pct, coh_idx)
    coh_pct_map = {c: p for c, p in zip(_COHERENCE_CENTERED, coherence_list)}
    coh_idx_map = {c: i for i, c in enumerate(_COHERENCE_CENTERED)}

    # Subject index for reproducible per-subject seeding
    subjects   = sorted(df_raw["subID"].unique())
    sub_to_idx = {s: i for i, s in enumerate(subjects)}

    rng  = np.random.default_rng(seed)
    rows = []

    for global_idx, (_, raw_row) in enumerate(df_raw.iterrows()):
        coh_centered = float(raw_row["coherence"])
        coh_pct      = coh_pct_map[coh_centered]
        coh_idx      = coh_idx_map[coh_centered]
        sub_idx      = sub_to_idx[raw_row["subID"]]

        # Deterministic direction per trial (seeded by global row position)
        direction       = int(rng.choice(DIRECTIONS))
        direction_label = DIR_TO_LABEL[direction]

        correct      = int(raw_row["correct"])
        human_choice = direction_label if correct == 1 else (1 - direction_label)
        rt_abs       = float(raw_row["rt"])

        # video_seed == global_idx: unique per row, always within [0, 2**32)
        video_seed = global_idx

        rows.append({
            "video_index":        global_idx,
            "subID":              str(raw_row["subID"]),
            "sub_idx":            sub_idx,
            "trial":              int(raw_row["trial"]),
            "dayTrials":          int(raw_row["dayTrials"]),
            "isVGP":              float(raw_row["isVGP"]),
            "coherence_idx":      coh_idx,
            "coherence":          coh_pct,
            "coherence_str":      f"{coh_pct}%",
            "coherence_centered": coh_centered,
            "direction":          direction,
            "direction_label":    direction_label,
            "rt":                 rt_abs,
            "correct":            correct,
            "human_choice":       human_choice,
            "video_seed":         video_seed,
        })

    df = pd.DataFrame(rows)

    # Assign stratified split
    split_masks = _stratified_split(df, val_size=val_size,
                                    test_size=test_size, seed=seed)
    df["split"] = "train"
    df.loc[split_masks["val"],  "split"] = "val"
    df.loc[split_masks["test"], "split"] = "test"

    print(
        f"Dataset built: {len(df)} total samples — "
        f"train {(df['split']=='train').sum()}, "
        f"val {(df['split']=='val').sum()}, "
        f"test {(df['split']=='test').sum()}."
    )
    print("  Accuracy by coherence:")
    for coh_idx, grp in df.groupby("coherence_idx"):
        coh_str = grp["coherence_str"].iloc[0]
        acc = grp["correct"].mean()
        print(f"    {coh_str:7s}  acc={acc:.3f}")

    if pkl_out is not None:
        pkl_out = Path(pkl_out)
        pkl_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(str(pkl_out))
        print(f"Metadata saved → {display_path(pkl_out)}")

    return RDMDataset(df=df)


# ── Fast reload ───────────────────────────────────────────────────────────────

def load_rdm_nrm_dataset(
    pkl_out:         Path | str            = PKL_OUT,
    embeddings_path: "Path | str | None"  = None,
) -> RDMDataset:
    """
    Fast reload of a previously built RDMDataset from the saved metadata pkl.

    Parameters
    ----------
    pkl_out         : path to the metadata pkl
    embeddings_path : if given, loads the motion-energy embeddings from a
                      single .pt file and attaches them
    """

    pkl_out = Path(pkl_out)
    print(f"Loading RDM metadata from {display_path(pkl_out)} ...")
    df = pd.read_pickle(str(pkl_out))

    if "human_choice" not in df.columns and "choice" in df.columns:
        df["human_choice"] = df["choice"]

    embeddings = None
    if embeddings_path is not None:
        embeddings_path = Path(embeddings_path)
        print(f"Loading embeddings from {display_path(embeddings_path)} ...")
        data       = torch.load(embeddings_path, weights_only=False)
        embeddings = data["embeddings"]
        if "video_indices" in data:
            idx_map    = {int(v): i for i, v in enumerate(data["video_indices"])}
            order      = [idx_map[int(vi)] for vi in df["video_index"]]
            embeddings = embeddings[order]

    return RDMDataset(df=df, embeddings=embeddings)


# ── Collate function ──────────────────────────────────────────────────────────

def rdm_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate for RDM DataLoader.

    - `choices`       : direction_label (ground-truth) — used in training loss
    - `human_choices` : what the human actually responded — used in evaluation
    - `coherences`    : coherence in % divided by 100 → [0, 1]
    - `image_index`   : alias for video_index, expected by scripts/eval_rdm.py
    """
    has_embeddings = "embedding" in batch[0]

    _vidx = torch.stack([b["video_index"] for b in batch])
    collated = {
        "video_index":  _vidx,
        "image_index":  _vidx,          # alias expected by scripts/eval_rdm.py
        "rts":          torch.stack([b["rt"]           for b in batch]),
        "coherences":   torch.stack([b["coherence"] / 100.0 for b in batch]),
        "human_choices": torch.stack([b["human_choice"] for b in batch]),
        "choices":       torch.stack([b["choice"]       for b in batch]),
        "levels":        [b["level_str"] for b in batch],
        "sub_ids":       [b["sub_id"]   for b in batch],
    }

    if has_embeddings:
        collated["embeddings"] = torch.stack([b["embedding"] for b in batch])

    return collated


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RDM dataset utilities")
    parser.add_argument("--sanity-check",          action="store_true")
    parser.add_argument("--generate-videos",        action="store_true")
    parser.add_argument("--compute-me-embeddings",   action="store_true",
                        help="Compute Motion Energy embeddings (Adelson & Bergen)")
    parser.add_argument("--rebuild",                 action="store_true",
                        help="Force rebuild from CSV even if pkl exists")
    parser.add_argument("--me-embeddings-path", type=str,
                        default=str(ME_EMBS_OUT))
    parser.add_argument("--device",            type=str, default=None)
    parser.add_argument("--n-directions",      type=int, default=8,
                        help="Motion energy: number of directions (default 8)")
    args = parser.parse_args()

    if PKL_OUT.exists() and not args.rebuild:
        dataset = load_rdm_nrm_dataset(pkl_out=PKL_OUT)
    else:
        dataset = build_rdm_nrm_dataset()

    if args.generate_videos:
        print("=== Generating RDM videos ===")
        dataset.generate_and_save_videos()

    elif args.compute_me_embeddings:
        print("=== Computing Motion Energy embeddings ===")
        dataset.compute_motion_energy_embeddings(
            save_path=args.me_embeddings_path,
            video_dir=VIDEO_DIR,
            device=args.device,
            n_directions=args.n_directions,
        )

    else:
        # Sanity check
        print("=== RDMDataset sanity check ===")
        dataset.remove_rt_outliers_mad()
        train_ds, val_ds, test_ds = dataset.train_val_test_split()

        loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                            collate_fn=rdm_collate_fn)
        batch = next(iter(loader))
        print("\nBatch keys:", list(batch.keys()))
        print("  rts:       ", batch["rts"].shape,        batch["rts"][:4])
        print("  coherences:", batch["coherences"].shape,  batch["coherences"][:4])
        print("  choices:   ", batch["choices"].shape,     batch["choices"][:4])
        print("  h_choices: ", batch["human_choices"].shape, batch["human_choices"][:4])
        print("  levels:    ", batch["levels"][:4])
        print("\nAll OK.")
