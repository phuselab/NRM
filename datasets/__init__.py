from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def display_path(path) -> str:
    """``path`` relative to the repository root when it lies inside it."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(_REPO_ROOT):
        return str(resolved.relative_to(_REPO_ROOT))
    return str(path)
