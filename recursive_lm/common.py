from pathlib import Path

def get_base_dir() -> Path:
    # Return the project root directory (parent of the recursive_lm package).
    return Path(__file__).resolve().parent.parent

def print_banner() -> None:
    banner_path = get_base_dir() / "banner.txt"
    print(banner_path.read_text(encoding="utf-8").rstrip(), flush=True)
