#!/usr/bin/env python3
"""Fetch everything the repository does not store in git, into its canonical path.

The git repository holds the code, the five NRM checkpoints, the SC embeddings
and the reference outputs. Everything else comes from one of three places:

**release** — files we produced that are too large for git (the RDM and
SAT stimulus embeddings), attached to a GitHub Release of this repository.

**source** — third-party data, checkpoints and code, downloaded from where
their authors publish them, under their own licences, and checked against the
sha256 of the files the paper used:

  * RDM behaviour — Zenodo 10.5281/zenodo.7025263 (CC-BY 4.0)
  * SAT behaviour — OSF osf.io/2cpmb, SATBench human data (CC-BY 4.0)
  * SC behaviour + stimuli — PLoS Comput Biol S1 File, Sofer et al. 2015 (CC-BY 4.0)
  * ConvRNN ImageNet backbone — OSF, rcnn-sat ``bl_imagenet.h5``
  * SCAN code — SATBench's ``scan/`` at a pinned commit

**built** — image sets whose source may not be redistributed, rebuilt locally:

  * SAT stimulus images — from your own copy of the ImageNet (ILSVRC2012)
    validation set, with ``datasets/sat/generate_images.py``
  * ``datasets/sc/sun_filtered/`` — from SUN397 on the Hugging Face Hub, with
    ``datasets/sc/build_sun_filtered.py`` (~40 GB download)

The competitor weights are not distributed: retrain them with
``competitors/*/train_*.py``. NRM itself needs only the git content plus the
``release`` assets.

    python fetch_assets.py --list
    python fetch_assets.py --all                 # release + source assets
    python fetch_assets.py embeddings-rdm sc-data
    python fetch_assets.py sat-images    # ImageNet in datasets/sat/ILSVRC/, or
    python fetch_assets.py sat-images --imagenet-val /path/to/ILSVRC/Data/CLS-LOC/val
    python fetch_assets.py sat-images --kaggle   # just its 1,100 images, from Kaggle
    python fetch_assets.py sun-filtered
    python fetch_assets.py --check               # report presence only

Existing files are never overwritten unless ``--force`` is given; downloads go
to a temporary file first and are moved into place only after their checksum
matches. ``--root DIR`` writes into DIR instead of the repository (useful to
test the whole chain without touching a working tree).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
#: Where sat-images looks for your ImageNet copy (git-ignored; not redistributable).
IMAGENET = REPO / "datasets/sat/ILSVRC"
#: The ImageNet files the SAT stimuli are made from (see generate_images.py).
SAT_SOURCES = REPO / "datasets/sat/imagenet_sources.csv"
#: Kaggle competition that serves ILSVRC2012 (`sat-images --kaggle`).
KAGGLE_COMPETITION = "imagenet-object-localization-challenge"
KAGGLE_VAL_DIR = "ILSVRC/Data/CLS-LOC/val"

#: GitHub repository and Release tag holding the large assets.
GH_REPO = os.environ.get("NRM_GH_REPO", "phuselab/NRM")
GH_TAG = os.environ.get("NRM_ASSETS_TAG", "assets-v1")

SATBENCH_COMMIT = "847c57811a67c2eb2f8424a1974f6b4078bb340f"


class Asset:
    kind = ""

    def __init__(self, name, files, approx_mb, note="", needed_for=""):
        self.name = name
        self.files = files            # {relative destination: sha256 or None}
        self.approx_mb = approx_mb
        self.note = note
        self.needed_for = needed_for

    def present(self, root):
        return all((root / f).exists() for f in self.files)


class Release(Asset):
    """A file attached to the GitHub Release ``GH_TAG`` of ``GH_REPO``."""
    kind = "release"


class Source(Asset):
    """Downloaded from its original publisher.

    ``url`` is a single file, or a zip archive when ``members`` maps archive
    paths to destinations (a trailing ``/`` unpacks a nested zip there).
    """
    kind = "source"

    def __init__(self, name, url, files, approx_mb, members=None, tree=None,
                 citation="", licence="", **kw):
        super().__init__(name, files, approx_mb, **kw)
        self.url = url
        self.members = members
        self.tree = tree              # (directory, digest of its .jpg files)
        self.citation = citation
        self.licence = licence

    def present(self, root):
        ok = super().present(root)
        if self.tree:
            d = root / self.tree[0]
            ok = ok and d.is_dir() and any(d.rglob("*.jpg"))
        return ok


class Built(Asset):
    """Rebuilt locally by a script from a source we may not redistribute."""
    kind = "built"

    def __init__(self, name, dest, approx_mb, expected_files, **kw):
        super().__init__(name, {}, approx_mb, **kw)
        self.dest = dest
        self.expected_files = expected_files

    def present(self, root):
        d = root / self.dest
        return d.is_dir() and sum(1 for p in d.rglob("*") if p.is_file()) >= self.expected_files


ASSETS = [
    Release("embeddings-rdm", {
        "embeddings/rdm/rdm_embeddings.pt":
            "b579853bf99b78346dfc27152207630587159f3d38f71b58b6c0a645985e8b2c"},
        803, note="motion-energy features of the synthetic RDM stimuli",
        needed_for="NRM on RDM"),
    Release("embeddings-sat", {
        "embeddings/sat/sat_color_blur_embeddings.pt":
            "8853180237bf84c45d7d9939645a2a944bbc41e048bbb0cde9d50de881bc5c19",
        "embeddings/sat/sat_color_gray_embeddings.pt":
            "a8069e966f95b765d3a68ac2f08da2c6cc38199a8ed0e2952b56094efda61c6d",
        "embeddings/sat/sat_gray_noise_embeddings.pt":
            "bc1c7e0caad9d67058c633553355ec68cf4020566552cd3158cf9352e7c2770b"},
        239, note="DINOv2 features, 3 perturbation conditions",
        needed_for="NRM on SAT"),
    Source("rdm-data",
           "https://zenodo.org/records/7025263/files/teddm_data.csv?download=1",
           {"datasets/rdm/teddm_data.csv":
                "88bce422a463c4587db85fd4b448beff8acc6f4c68e3fd7c5584cb23ee4f791c"},
           3, licence="CC-BY 4.0",
           citation="Cochrane, Sims, Bejjanki, Green & Bavelier 2023, npj Science "
                    "of Learning 8, doi:10.1038/s41539-023-00168-9; data "
                    "doi:10.5281/zenodo.7025263",
           note="58,438 dot-motion trials", needed_for="RDM evaluation"),
    Source("sat-data", "https://osf.io/download/bw3mc/",
           {"datasets/sat/color-blur.txt":
                "7cf77a31c5be28608e3f39745f75ba5408362b1d021d5adb2bd30c889e4da333",
            "datasets/sat/color-gray.txt":
                "2df49f575169f83d3a03decd991abaa4c5093c653c81fe39490eea02958a2db0",
            "datasets/sat/gray-noise.txt":
                "b3f3ec795ee928d7e3c49c1022740ad2d85f5ea8f61c771482018b1e2b86585a"},
           38, members={
               "human-data/main/color-blur.txt": "datasets/sat/color-blur.txt",
               "human-data/main/color-gray.txt": "datasets/sat/color-gray.txt",
               "human-data/main/gray-noise.txt": "datasets/sat/gray-noise.txt"},
           licence="CC-BY 4.0",
           citation="Subramanian et al. 2022, SATBench, osf.io/2cpmb",
           note="SATBench human data (human-data.zip)", needed_for="SAT evaluation"),
    Source("sc-data",
           "https://journals.plos.org/ploscompbiol/article/file?type=supplementary"
           "&id=10.1371/journal.pcbi.1004456.s002",
           {"datasets/sc/experiment_1.csv":
                "6bb2a65530abdffeff9717af2e41e9b2df4b89310b4d77986ac33581ab5fe9dc"},
           28, members={
               "Sofer SI/data/experiment_1.csv": "datasets/sc/experiment_1.csv",
               "Sofer SI/stimuli/experiment_1_stimuli.zip": "datasets/sc/stimuli/",
           },
           tree=("datasets/sc/stimuli",
                 "7e66c6dfebfc364c0b25b1b5f618399b2259f1cfaeaa7132f3d88399e84afe4c"),
           licence="CC-BY 4.0",
           citation="Sofer, Crouzet & Serre 2015, PLoS Comput Biol 11(9):e1004456, "
                    "S1 File",
           note="behaviour of experiment 1 and its 960 stimuli",
           needed_for="SC evaluation; recomputing SC embeddings"),
    Source("convrnn-backbone", "https://osf.io/9td5p/download",
           {"competitors/weights/sat/convrnn_backbone.h5":
                "7584b1430c4a74b5e5d00b522a133069ee7d1740c0471c39a9cea9bc4e93f381"},
           112, licence="released by the authors (rcnn-sat, MIT)",
           citation="Spoerer, Kietzmann, Mehrer, Charest & Kriegeskorte 2020",
           note="ImageNet-trained BL-Net (bl_imagenet.h5)",
           needed_for="ConvRNN training on SAT"),
    Source("scan-code",
           f"https://codeload.github.com/ajaysub110/satbench/zip/{SATBENCH_COMMIT}",
           {"competitors/scan/sresnet.py":
                "a56b010569e0365ccdf5b9ab83a4862a84013daaf43bf7eccf454a83e6194376",
            "competitors/scan/small_resnet.py":
                "b3ea211f1cd06986e47a15e2a6cafd61c7f0151f8d1a8a4918df819cc77ed4ae",
            "competitors/scan/train.py":
                "1de69e289ea3cc71d99a530abc20a83d8650bff593e6c50ae3e9da2a4f3a6a52",
            "competitors/scan/test.py":
                "3ce368e1e46a0592700ff11c72fa1d3c9fae60970ba57e4cd2df9c0b9c87bca2",
            "competitors/scan/utils.py":
                "aa2e55650af48b835c28f46ee2ecd939baa0e8bb39bc86453aa34de5ec267d11",
            "competitors/scan/custom_transforms.py":
                "88fed9002b1c8366df0f7d1a7fcdd4de10a9da246b672a9dfb410b6d8f71b8e5",
            "competitors/scan/requirements.txt":
                "4af754393c5c2350ed499aa8ab5ea4d63da89c9e13f95420a0888b932df76c96"},
           35, members={
               f"satbench-{SATBENCH_COMMIT}/scan/{f}": f"competitors/scan/{f}"
               for f in ("sresnet.py", "small_resnet.py", "train.py", "test.py",
                         "utils.py", "custom_transforms.py", "requirements.txt")},
           licence="no licence published upstream — fetched, not redistributed",
           citation="Zhang et al. 2019 (SCAN), as adapted by SATBench",
           needed_for="SCAN on SAT"),
    Built("sat-images", "datasets/sat/data", 186, expected_files=8800,
          note="from 1,100 ImageNet validation images; see --imagenet-val",
          needed_for="recomputing SAT embeddings; image-based SAT competitors"),
    Built("sun-filtered", "datasets/sc/sun_filtered", 111, expected_files=8762,
          note="from SUN397 on the Hugging Face Hub (~40 GB download); needs "
               "pyarrow (uv sync --extra data)",
          needed_for="RTify-SC / cRNN pre-training"),
]

BY_NAME = {a.name: a for a in ASSETS}
#: What --all fetches: everything that downloads without credentials or a
#: multi-GB source.
DEFAULT = [a.name for a in ASSETS if a.kind in ("release", "source")]


# ── helpers ──────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(directory: Path) -> str:
    """sha256 over the sorted (sha256, path) list of the .jpg files below."""
    lines = []
    for p in sorted(directory.rglob("*.jpg"), key=lambda p: str(p.relative_to(directory))):
        lines.append(f"{sha256(p)}  ./{p.relative_to(directory)}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def download(url: str, dest: Path, headers=None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "nrm-fetch-assets",
                                               **(headers or {})})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def place(tmp: Path, dest: Path, expected: str | None, force: bool) -> None:
    if expected and sha256(tmp) != expected:
        raise RuntimeError(f"checksum mismatch for {dest.name}: the source no "
                           "longer serves the file the paper used")
    if dest.exists() and not force:
        print(f"    kept existing {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), dest)
    print(f"    -> {dest}")


def release_download(fname: str, dest: Path) -> None:
    """Public Release URL first; the REST API with GITHUB_TOKEN for a private repo."""
    url = f"https://github.com/{GH_REPO}/releases/download/{GH_TAG}/{fname}"
    try:
        download(url, dest)
        return
    except urllib.error.HTTPError as e:
        token = os.environ.get("GITHUB_TOKEN")
        if e.code != 404 or not token:
            raise RuntimeError(
                f"{url} -> HTTP {e.code}. If the repository is private, set "
                "GITHUB_TOKEN to a token with read access.") from e
    auth = {"Authorization": f"Bearer {token}"}
    api = f"https://api.github.com/repos/{GH_REPO}/releases/tags/{GH_TAG}"
    with urllib.request.urlopen(urllib.request.Request(api, headers=auth)) as r:
        assets = {a["name"]: a["url"] for a in json.load(r)["assets"]}
    if fname not in assets:
        raise RuntimeError(f"{fname} is not attached to release {GH_TAG}")
    download(assets[fname], dest, {**auth, "Accept": "application/octet-stream"})


# ── fetchers ─────────────────────────────────────────────────────────────────

def fetch_release(a: Release, root: Path, force: bool) -> None:
    for rel, expected in a.files.items():
        dest = root / rel
        if dest.exists() and not force:
            print(f"    present {rel}")
            continue
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / Path(rel).name
            release_download(Path(rel).name, tmp)
            place(tmp, dest, expected, force)


def fetch_source(a: Source, root: Path, force: bool) -> None:
    if a.present(root) and not force:
        print("    present")
        return
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "download"
        print(f"    downloading {a.url}")
        download(a.url, tmp)
        if a.members is None:
            (rel, expected), = a.files.items()
            place(tmp, root / rel, expected, force)
            return
        with zipfile.ZipFile(tmp) as z:
            for member, rel in a.members.items():
                if rel.endswith("/"):
                    # a nested zip, unpacked into this directory
                    dest_dir = root / rel
                    with zipfile.ZipFile(io.BytesIO(z.read(member))) as inner:
                        for info in inner.infolist():
                            name = info.filename
                            if info.is_dir() or "__MACOSX" in name or \
                                    Path(name).name.startswith("."):
                                continue
                            out = dest_dir / name
                            if out.exists() and not force:
                                continue
                            out.parent.mkdir(parents=True, exist_ok=True)
                            out.write_bytes(inner.read(info))
                    print(f"    -> {dest_dir}")
                else:
                    part = Path(td) / Path(rel).name
                    part.write_bytes(z.read(member))
                    place(part, root / rel, a.files.get(rel), force)
    if a.tree:
        d = root / a.tree[0]
        if tree_digest(d) != a.tree[1]:
            raise RuntimeError(f"{d}: contents differ from the files the paper used")


def kaggle_api(no_browser: bool):
    """An authenticated KaggleApi, running `kaggle auth login` when needed."""
    if importlib.util.find_spec("kaggle") is None:
        raise RuntimeError("--kaggle needs the kaggle package (Python >= 3.11): run "
                           "`uv run --extra kaggle python fetch_assets.py sat-images --kaggle`.")
    import contextlib
    # The kaggle package authenticates on import, printing help when there are
    # no credentials, and on failure calls site's exit(), which closes stdin.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        kae = importlib.import_module("kaggle.api.kaggle_api_extended")
    kae.exit = sys.exit

    def authenticated():
        api = kae.KaggleApi()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                api.authenticate()
            return api
        except SystemExit:          # no credentials found
            return None

    api = authenticated()
    if api is None:
        print("    No Kaggle credentials found: logging in (OAuth, in your browser).\n"
              "    No account yet? Sign up at https://www.kaggle.com/account/login",
              flush=True)
        # The CLI entry point, in a process of its own: it needs a stdin, and
        # only when run as `kaggle` does the package let `auth login` through.
        cli = Path(sys.executable).with_name("kaggle")
        cmd = [str(cli) if cli.exists() else "kaggle", "auth", "login"]
        if no_browser:
            cmd.append("--no-launch-browser")
        try:
            subprocess.run(cmd, stdin=0, check=False)
        except KeyboardInterrupt:
            raise RuntimeError("Kaggle login aborted.") from None
        api = authenticated()
        if api is None:
            raise RuntimeError("Kaggle login failed; rerun to try again "
                               "(--kaggle-no-browser to paste a code instead).")
    return api


def kaggle_download(api, file_name: str, path: str, bar, waits=(30, 60, 120, 300, 600)):
    """One competition file, waiting out Kaggle's rate limit (HTTP 429).

    The kaggle package retries a 429 only while streaming the file, not on the
    request that starts the download, which is where the limit bites.
    """
    import time
    from requests.exceptions import HTTPError

    for attempt, wait in enumerate((*waits, None)):
        try:
            api.competition_download_file(KAGGLE_COMPETITION, file_name,
                                          path=path, force=True, quiet=True)
            return
        except HTTPError as e:
            r = e.response
            if r is None or r.status_code != 429 or wait is None:
                raise
            after = r.headers.get("Retry-After", "")
            wait = int(after) if after.isdigit() else wait
            bar.set_postfix_str(f"rate-limited, waiting {wait}s")
            time.sleep(wait)
            bar.set_postfix_str("")


def fetch_kaggle_sources(val: Path, needed: list[str], no_browser: bool) -> None:
    """Download the listed ILSVRC2012 validation images from Kaggle into val."""
    missing = [fn for fn in needed if not (val / fn).is_file()]
    if not missing:
        return
    api = kaggle_api(no_browser)
    from requests.exceptions import HTTPError
    from tqdm import tqdm

    val.mkdir(parents=True, exist_ok=True)
    print(f"    downloading {len(missing)} images from Kaggle ({KAGGLE_COMPETITION}) "
          f"into {val}")
    with tempfile.TemporaryDirectory() as td:
        bar = tqdm(missing, unit="img", desc="    ImageNet")
        for fn in bar:
            try:
                kaggle_download(api, f"{KAGGLE_VAL_DIR}/{fn}", td, bar)
            except HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                if code == 403:
                    raise RuntimeError(
                        "Kaggle refused the download (HTTP 403): accept the "
                        "competition rules once, logged in as you, then rerun:\n"
                        f"      https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules"
                    ) from None
                if code == 401:
                    raise RuntimeError(
                        "Kaggle rejected the credentials (HTTP 401): run "
                        "`uv run --extra kaggle kaggle auth login --force`, then rerun.") from None
                raise RuntimeError(f"Kaggle download of {fn} failed: {e}") from None
            # Served as is or zipped, depending on the file.
            got = Path(td) / fn
            if not got.is_file() and (Path(td) / f"{fn}.zip").is_file():
                with zipfile.ZipFile(Path(td) / f"{fn}.zip") as z:
                    got.write_bytes(z.read(Path(z.namelist()[0]).as_posix()))
            if not got.is_file():
                raise RuntimeError(f"Kaggle returned no file for {fn} "
                                   f"(got: {sorted(os.listdir(td))})")
            if got.read_bytes()[:2] != b"\xff\xd8":
                raise RuntimeError(f"{fn} from Kaggle is not a JPEG")
            shutil.move(str(got), val / fn)
            for leftover in Path(td).iterdir():
                leftover.unlink()


def fetch_built(a: Built, root: Path, args) -> None:
    if a.present(root) and not args.force:
        print("    present")
        return
    py = sys.executable
    if a.name == "sat-images":
        # The 1,100 validation images the stimuli are made from, listed in
        # datasets/sat/imagenet_sources.csv (no labels CSV needed).
        val = Path(args.imagenet_val)
        with open(SAT_SOURCES) as f:
            needed = [line.split(",")[2] for line in list(f)[1:]]
        if args.kaggle:
            fetch_kaggle_sources(val, needed, args.kaggle_no_browser)
        missing = [fn for fn in needed if not (val / fn).is_file()]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of the {len(needed)} ImageNet validation images "
                f"listed in datasets/sat/imagenet_sources.csv are not in {val} "
                f"(first: {missing[0]}). sat-images needs your own copy of the "
                "ILSVRC2012 validation set (e.g. from the Kaggle 'ImageNet Object "
                "Localization Challenge'): put it at datasets/sat/ILSVRC/Data/CLS-LOC/val, "
                "pass --imagenet-val, or download just these images with --kaggle.")
        cmd = [py, str(REPO / "datasets/sat/generate_images.py"),
               "--val_dir", str(val), "--output_dir", str(root / a.dest)]
    else:
        # checked here, not after the ~40 GB download
        missing = [m for m in ("pyarrow", "huggingface_hub")
                   if importlib.util.find_spec(m) is None]
        if missing:
            raise RuntimeError(
                f"sun-filtered needs {', '.join(missing)}: run `uv sync --extra data` "
                "first, or `uv run --extra data python fetch_assets.py sun-filtered`.")
        cmd = [py, str(REPO / "datasets/sc/build_sun_filtered.py"),
               "--output_dir", str(root / a.dest),
               "--exp1_csv", str(root / "datasets/sc/experiment_1.csv")]
    print("    " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch(name: str, root: Path, args) -> bool:
    a = BY_NAME[name]
    print(f"[{a.kind}] {name}")
    try:
        if a.kind == "release":
            fetch_release(a, root, args.force)
        elif a.kind == "source":
            fetch_source(a, root, args.force)
        else:
            fetch_built(a, root, args)
        return True
    except Exception as e:      # report and carry on with the other assets
        print(f"    FAILED: {e}")
        return False


# ── CLI ──────────────────────────────────────────────────────────────────────

def show(root: Path) -> None:
    print(f"  {'asset':<18}{'kind':<9}{'size':>8}  {'status':<8} needed for")
    for a in ASSETS:
        status = "present" if a.present(root) else "absent"
        print(f"  {a.name:<18}{a.kind:<9}{a.approx_mb:>5} MB  {status:<8} {a.needed_for}")
    print("\nSources:")
    for a in ASSETS:
        if isinstance(a, Source):
            print(f"  {a.name:<18}{a.citation}  [{a.licence}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help=f"assets to fetch: {', '.join(BY_NAME)}")
    ap.add_argument("--all", action="store_true",
                    help="every release and source asset (not the rebuilt image sets)")
    ap.add_argument("--list", action="store_true", help="show assets and their status")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if a default asset is absent")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--root", default=str(REPO),
                    help="write into this directory instead of the repository")
    ap.add_argument("--imagenet-val", default=str(IMAGENET / "Data/CLS-LOC/val"),
                    help="ILSVRC2012 CLS-LOC/val directory "
                         "(default: datasets/sat/ILSVRC/Data/CLS-LOC/val)")
    ap.add_argument("--kaggle", action="store_true",
                    help="sat-images: download the 1,100 images it needs (~140 MB) from "
                         "Kaggle into --imagenet-val (needs `--extra kaggle`, a Kaggle "
                         "account and the competition rules accepted)")
    ap.add_argument("--kaggle-no-browser", action="store_true",
                    help="log in to Kaggle by pasting a code instead of a browser "
                         "callback (for machines without a browser, e.g. over SSH)")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    if args.list or not (args.names or args.all or args.check):
        show(root)
        return
    if args.check:
        missing = [n for n in DEFAULT if not BY_NAME[n].present(root)]
        show(root)
        sys.exit(1 if missing else 0)

    names = DEFAULT if args.all else args.names
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        ap.error(f"unknown asset(s): {', '.join(unknown)}")
    ok = [fetch(n, root, args) for n in names]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
