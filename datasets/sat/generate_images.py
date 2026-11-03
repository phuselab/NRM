#!/usr/bin/env python3
"""Rebuild SAT stimulus images from the ImageNet (ILSVRC2012) validation set."""

import argparse
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# 16-category WordNet mapping (SATBench ImagenetCOCOMapping.py)
# ---------------------------------------------------------------------------
MAPPINGS = {
    "knife":    ["n03041632"],
    "keyboard": ["n03085013", "n04505470"],
    "elephant": ["n02504013", "n02504458"],
    "bicycle":  ["n02835271", "n03792782"],
    "airplane": ["n02690373", "n03955296", "n13861050", "n13941806"],
    "clock":    ["n02708093", "n03196217", "n04548280"],
    "oven":     ["n03259401", "n04111414", "n04111531"],
    "chair":    ["n02791124", "n03376595", "n04099969", "n00605023", "n04429376"],
    "bear":     ["n02132136", "n02133161", "n02134084", "n02134418"],
    "boat":     ["n02951358", "n03344393", "n03662601", "n04273569", "n04612373", "n04612504"],
    "cat":      ["n02122878", "n02123045", "n02123159", "n02126465", "n02123394", "n02123597",
                 "n02124075", "n02125311"],
    "bottle":   ["n02823428", "n03937543", "n03983396", "n04557648", "n04560804", "n04579145",
                 "n04591713"],
    "truck":    ["n03345487", "n03417042", "n03770679", "n03796401", "n00319176", "n01016201",
                 "n03930630", "n03930777", "n05061003", "n06547832", "n10432053", "n03977966",
                 "n04461696", "n04467665"],
    "car":      ["n02814533", "n03100240", "n03100346", "n13419325", "n04285008"],
    "bird":     ["n01321123", "n01514859", "n01792640", "n07646067", "n01530575", "n01531178",
                 "n01532829", "n01534433", "n01537544", "n01558993", "n01562265", "n01560419",
                 "n01582220", "n10281276", "n01592084", "n01601694", "n01614925", "n01616318",
                 "n01622779", "n01795545", "n01796340", "n01797886", "n01798484", "n01817953",
                 "n01818515", "n01819313", "n01820546", "n01824575", "n01828970", "n01829413",
                 "n01833805", "n01843065", "n01843383", "n01855032", "n01855672", "n07646821",
                 "n01860187", "n02002556", "n02002724", "n02006656", "n02007558", "n02009229",
                 "n02009912", "n02011460", "n02013706", "n02017213", "n02018207", "n02018795",
                 "n02025239", "n02027492", "n02028035", "n02033041", "n02037110", "n02051845",
                 "n02056570"],
    "dog":      ["n02085782", "n02085936", "n02086079", "n02086240", "n02086646", "n02086910",
                 "n02087046", "n02087394", "n02088094", "n02088238", "n02088364", "n02088466",
                 "n02088632", "n02089078", "n02089867", "n02089973", "n02090379", "n02090622",
                 "n02090721", "n02091032", "n02091134", "n02091244", "n02091467", "n02091635",
                 "n02091831", "n02092002", "n02092339", "n02093256", "n02093428", "n02093647",
                 "n02093754", "n02093859", "n02093991", "n02094114", "n02094258", "n02094433",
                 "n02095314", "n02095570", "n02095889", "n02096051", "n02096294", "n02096437",
                 "n02096585", "n02097047", "n02097130", "n02097209", "n02097298", "n02097474",
                 "n02097658", "n02098105", "n02098286", "n02099267", "n02099429", "n02099601",
                 "n02099712", "n02099849", "n02100236", "n02100583", "n02100735", "n02100877",
                 "n02101006", "n02101388", "n02101556", "n02102040", "n02102177", "n02102318",
                 "n02102480", "n02102973", "n02104029", "n02104365", "n02105056", "n02105162",
                 "n02105251", "n02105505", "n02105641", "n02105855", "n02106030", "n02106166",
                 "n02106382", "n02106550", "n02106662", "n02107142", "n02107312", "n02107574",
                 "n02107683", "n02107908", "n02108000", "n02108422", "n02108551", "n02108915",
                 "n02109047", "n02109525", "n02109961", "n02110063", "n02110185", "n02110627",
                 "n02110806", "n02110958", "n02111129", "n02111277", "n08825211", "n02111500",
                 "n02112018", "n02112350", "n02112706", "n02113023", "n02113624", "n02113712",
                 "n02113799", "n02113978"],
}

# ---------------------------------------------------------------------------
# Perturbation transforms (as in SATBench create_humanSplits.py)
# ---------------------------------------------------------------------------

class _AddGaussianBlur:
    def __init__(self, kernel=7, std=0.0):
        self.kernel = kernel
        self.std    = std

    def __call__(self, tensor):
        if self.std != 0.0:
            tensor = transforms.GaussianBlur(kernel_size=self.kernel, sigma=self.std)(tensor)
        return tensor


class _AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0, contrast=0.1):
        self.std      = std
        self.mean     = mean
        self.contrast = contrast

    def __call__(self, tensor):
        n   = tensor.size(1) * tensor.size(2)
        sd2 = self.std * 2
        noise = torch.Tensor()
        while len(noise) < n:
            m   = 2 * (n - len(noise))
            new = torch.randn(m) * self.std
            new = new[new >= -sd2]
            new = new[new <= sd2]
            noise = torch.cat([noise, new])
        noise    = torch.reshape(noise[:n], (tensor.size(1), tensor.size(2)))
        newnoise = torch.stack([noise, noise, noise]) + self.mean
        tensor   = tensor + (0.5 - tensor.mean())
        tensor   = transforms.functional.adjust_contrast(tensor, self.contrast)
        return tensor + newnoise + self.mean


def _blur_tf(sigma: float):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        _AddGaussianBlur(7, sigma),
        transforms.ToPILImage(),
    ])


def _noise_tf(sigma: float):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        _AddGaussianNoise(0.0, sigma),
        transforms.ToPILImage(),
    ])


def _color_gray_tf(mode: str):
    """mode='c' keeps colour; mode='g' converts to greyscale (3-channel)."""
    if mode == 'g':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.ToPILImage(),
        ])
    else:  # 'c'
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.ToPILImage(),
        ])


# All (perturbation_type, mode_value) pairs to generate per image slot.
# Covers every filename that appears in color-blur.txt, gray-noise.txt,
# and color-gray.txt.
VARIANTS = [
    # (type,       mode_value,  filename_mode_str)
    ('blur',       0,           '0'),
    ('blur',       3,           '3'),
    ('blur',       9,           '9'),
    ('noise',      0.0,         '0'),
    ('noise',      0.04,        '0.04'),
    ('noise',      0.16,        '0.16'),
    ('color_gray', 'c',         'c'),
    ('color_gray', 'g',         'g'),
]

# Pre-build all transforms (thread-safe, stateless)
_TRANSFORMS = {}
for _ptype, _mval, _mstr in VARIANTS:
    if _ptype == 'blur':
        _TRANSFORMS[(_ptype, _mstr)] = _blur_tf(float(_mval))
    elif _ptype == 'noise':
        _TRANSFORMS[(_ptype, _mstr)] = _noise_tf(float(_mval))
    elif _ptype == 'color_gray':
        _TRANSFORMS[(_ptype, _mstr)] = _color_gray_tf(_mval)


# ---------------------------------------------------------------------------
# Step 1 — create 16-class ImageNet directory
# ---------------------------------------------------------------------------

def create_16class(val_dir: str, label_csv: str, class16_dir: str) -> None:
    """
    Copies images from the flat val/ directory into per-category subdirectories.
    Same logic as create16ClassImagenet.py, but uses shutil for portability.
    """
    import pandas as pd

    # Build WordNet-ID → category name reverse mapping
    wnetid_to_cat = {
        wn: cat
        for cat, wnetids in MAPPINGS.items()
        for wn in wnetids
    }

    Path(class16_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(label_csv)

    copied = skipped_cat = skipped_src = 0
    for _, row in df.iterrows():
        filename = row['ImageId'] + '.JPEG'
        wnetid   = row['PredictionString'].split()[0]
        cat      = wnetid_to_cat.get(wnetid)
        if cat is None:
            skipped_cat += 1
            continue
        src = os.path.join(val_dir, filename)
        if not os.path.exists(src):
            skipped_src += 1
            continue
        cat_dir = os.path.join(class16_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        dst = os.path.join(cat_dir, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied += 1

    print(f"  16-class done: {copied} images, "
          f"{skipped_cat} no-category, {skipped_src} missing source.")


# ---------------------------------------------------------------------------
# Step 2 — build deterministic split list (flist)
# ---------------------------------------------------------------------------

def build_flist(class16_dir: str, n_modes: int, n_rts: int,
                n_cat_training: int = 50, seed: int = 42) -> dict:
    """
    Reproduces create_humanSplits.py's flist logic using sorted() file lists
    for full determinism regardless of filesystem.

    Returns
    -------
    dict: {'train': [(fn, cat), ...], '0': [...], '1': [...], ..., '4': [...]}
    where fn is the flat ILSVRC filename (e.g. 'ILSVRC2012_val_00007476.JPEG').
    """
    random.seed(seed)
    np.random.seed(seed)

    # n_samples_needed_per_mode = (1000 + 50) // (n_rts * n_modes) = 70
    n_samples_needed_per_mode    = (1000 + 50) // (n_rts * n_modes)
    # target per RT block = n_modes * 70 = 210
    target_per_rt                = n_modes * n_samples_needed_per_mode
    # samples per category for primary fill = 70*3*5//16 = 65
    n_samples_needed_per_category = n_samples_needed_per_mode * n_modes * n_rts // len(MAPPINGS)

    def sorted_files(cat):
        d = os.path.join(class16_dir, cat)
        return sorted(os.listdir(d)) if os.path.isdir(d) else []

    fcounts = {cat: len(sorted_files(cat)) for cat in MAPPINGS}

    flist = {str(rt): [] for rt in range(n_rts)}
    flist['train'] = []

    n_sampled_earlier   = {cat: 0 for cat in MAPPINGS}
    remaining_files     = []
    remaining_cat_counts = {}

    for cat in MAPPINGS:
        files       = sorted_files(cat)
        n_sampleable = min(fcounts[cat], n_samples_needed_per_category)
        n_per_rt     = n_sampleable // n_rts

        for rt in range(n_rts):
            start = n_sampled_earlier[cat]
            flist[str(rt)] += [(f, cat) for f in files[start: start + n_per_rt]]
            n_sampled_earlier[cat] += n_per_rt

        remaining_cat_counts[cat] = len(files[n_sampled_earlier[cat]:])
        remaining_files          += [(f, cat) for f in files[n_sampled_earlier[cat]:]]

    # Fill each RT block up to target_per_rt from remaining pool
    for rt in range(n_rts):
        deficit = target_per_rt - len(flist[str(rt)])
        if deficit > 0:
            available = min(deficit, len(remaining_files))
            flist[str(rt)] += random.sample(remaining_files, available)

    # Training set: 50 images sampled proportionally from remaining
    nonzero_cat = {k: v for k, v in remaining_cat_counts.items() if v > 0}
    if nonzero_cat and len(remaining_files) >= n_cat_training:
        prob_cat = 1.0 / len(nonzero_cat)
        probs    = np.array([prob_cat / nonzero_cat[fc[1]] for fc in remaining_files],
                            dtype=float)
        probs   /= probs.sum()
        inds     = np.random.choice(len(remaining_files), n_cat_training,
                                    replace=False, p=probs)
        flist['train'] = [remaining_files[i] for i in inds]

    sizes = {k: len(v) for k, v in flist.items()}
    print(f"  flist sizes: {sizes}")
    total = sum(sizes.values())
    print(f"  Total image slots: {total}  "
          f"(× {len(VARIANTS)} variants = {total * len(VARIANTS)} output files)")
    return flist


# ---------------------------------------------------------------------------
# The split list, stored: imagenet_sources.csv
# ---------------------------------------------------------------------------
# build_flist() picks from the full 16-class listing, so it needs all 50,000
# validation images to reproduce the selection. The CSV records its result
# (1,100 file names, no images), so a rebuild needs only those files.

def write_sources(flist: dict, path: str) -> None:
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['folder', 'slot', 'filename', 'category'])
        for folder, entries in flist.items():
            for j, (fn, cat) in enumerate(entries):
                w.writerow([folder, j, fn, cat])


def read_sources(path: str) -> dict:
    import csv
    flist = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            entries = flist.setdefault(row['folder'], [])
            if int(row['slot']) != len(entries):
                raise ValueError(f"{path}: slots of folder {row['folder']} out of order")
            entries.append((row['filename'], row['category']))
    return flist


# ---------------------------------------------------------------------------
# Step 3 — generate all image variants
# ---------------------------------------------------------------------------

def _process_slot(args_tuple):
    """Worker function: generate all perturbation variants for one image slot."""
    folder, j, src_fn, cat, val_dir, output_dir = args_tuple
    src_path = os.path.join(val_dir, src_fn)
    if not os.path.exists(src_path):
        return 0, 1  # (saved, missing_src)

    try:
        img = Image.open(src_path).convert('RGB')
    except Exception as e:
        print(f"\n  [WARN] Cannot open {src_path}: {e}")
        return 0, 1

    saved = 0
    for ptype, mval, mstr in VARIANTS:
        out_name = f"{j}_{ptype}_{mstr}_{cat}.JPEG"
        out_path = os.path.join(output_dir, folder, out_name)
        if os.path.exists(out_path):
            saved += 1
            continue
        try:
            out_img = _TRANSFORMS[(ptype, mstr)](img)
            out_img.save(out_path, 'JPEG', quality=95)
            saved += 1
        except Exception as e:
            print(f"\n  [WARN] Failed {out_path}: {e}")
    return saved, 0


def generate_images(flist: dict, val_dir: str, output_dir: str,
                    workers: int = 4) -> None:
    # Create output subfolders
    for folder in flist:
        Path(output_dir, folder).mkdir(parents=True, exist_ok=True)

    tasks = [
        (folder, j, src_fn, cat, val_dir, output_dir)
        for folder, entries in flist.items()
        for j, (src_fn, cat) in enumerate(entries)
    ]

    total_slots = len(tasks)
    print(f"  {total_slots} image slots, {workers} workers …")

    done = saved_total = missing_total = 0
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futs = {exe.submit(_process_slot, t): t for t in tasks}
        for fut in as_completed(futs):
            s, m = fut.result()
            saved_total   += s
            missing_total += m
            done += 1
            if done % 50 == 0 or done == total_slots:
                print(f"  {done}/{total_slots} slots … "
                      f"{saved_total} files written", end='\r')
    print()
    if missing_total:
        print(f"  [WARN] {missing_total} source images not found in val_dir.")


# ---------------------------------------------------------------------------
# Verification: check coverage for a behavioural .txt file
# ---------------------------------------------------------------------------

def verify_coverage(txt_path: str, output_dir: str) -> None:
    """Prints how many filenames in txt_path are present in output_dir."""
    import json
    import pandas as pd

    BLOCK_TO_FOLDER = {
        'cattut': 'train',
        '0': '0', '400': '1', '600': '2', '800': '3', '1000': '4',
    }
    BLOCKS = ['cattut', '0', '400', '600', '800', '1000']

    needed = set()
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sess = json.loads(line)
            except Exception:
                continue
            df = pd.DataFrame(sess)
            for block in BLOCKS:
                screen = 'CategorizationTutorial' if block == 'cattut' else block
                match  = df.loc[df['sender'] == 'Screen' + screen]
                if len(match) == 0:
                    continue
                start    = match.index[0]
                n_samp   = 50 * 3 if block == 'cattut' else 210 * 3
                n_skip   = 0      if block == 'cattut' else 10
                window   = df.iloc[start: start + n_samp + 1]
                block_df = window[window['sender'] == 'Stimulus & Response'].iloc[n_skip:]
                folder   = BLOCK_TO_FOLDER[block]
                for fn in block_df['filename'].dropna():
                    needed.add((folder, str(fn)))

    found = sum(
        1 for fld, fn in needed
        if os.path.exists(os.path.join(output_dir, fld, fn))
    )
    name = os.path.basename(txt_path)
    pct  = 100 * found / len(needed) if needed else 0.0
    print(f"  {name}: {found}/{len(needed)} images found ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild the SAT stimulus images from ImageNet validation.')
    parser.add_argument('--val_dir', default=str(_HERE / 'ILSVRC/Data/CLS-LOC/val'),
                        help='ILSVRC CLS-LOC/val flat directory '
                             '(default: datasets/sat/ILSVRC/Data/CLS-LOC/val)')
    parser.add_argument('--label_csv', default=str(_HERE / 'ILSVRC/LOC_val_solution.csv'),
                        help='LOC_val_solution.csv path '
                             '(default: datasets/sat/ILSVRC/LOC_val_solution.csv)')
    parser.add_argument('--output_dir', default=str(_HERE / 'data'),
                        help='Root output dir (default: datasets/sat/data)')
    parser.add_argument('--n_rts',    type=int, default=5)
    parser.add_argument('--n_modes',  type=int, default=3)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--sources', default=str(_HERE / 'imagenet_sources.csv'),
                        help='Stored split list; when it exists, steps 1-2 are skipped '
                             'and only its 1,100 source images are needed '
                             '(default: datasets/sat/imagenet_sources.csv)')
    parser.add_argument('--rebuild_sources', action='store_true',
                        help='Redo steps 1-2 from the full validation set and the '
                             'label CSV, and (re)write --sources')
    parser.add_argument('--skip_16class', action='store_true',
                        help='Skip step 1 if <output_dir>/16class already exists')
    parser.add_argument('--keep_16class', action='store_true',
                        help='Keep the intermediate <output_dir>/16class directory')
    args = parser.parse_args()

    class16_dir = os.path.join(args.output_dir, '16class')

    if os.path.exists(args.sources) and not args.rebuild_sources:
        print(f'\nSteps 1-2 — split list from {args.sources}')
        flist = read_sources(args.sources)
        print(f"  flist sizes: { {k: len(v) for k, v in flist.items()} }")
    else:
        print('\nStep 1 — 16-class ImageNet directory')
        if args.skip_16class and os.path.isdir(class16_dir):
            print(f'  Skipping (--skip_16class): {class16_dir}')
        else:
            create_16class(args.val_dir, args.label_csv, class16_dir)

        print('\nStep 2 — split list')
        flist = build_flist(class16_dir, n_modes=args.n_modes, n_rts=args.n_rts,
                            n_cat_training=50, seed=args.seed)
        write_sources(flist, args.sources)
        print(f'  -> {args.sources}')
        if not args.keep_16class:
            shutil.rmtree(class16_dir, ignore_errors=True)

    print('\nStep 3 — image variants')
    generate_images(flist, args.val_dir, args.output_dir, workers=args.workers)

    for txt in ('color-blur.txt', 'color-gray.txt', 'gray-noise.txt'):
        path = _HERE / txt
        if path.exists():
            verify_coverage(str(path), args.output_dir)


if __name__ == '__main__':
    main()
