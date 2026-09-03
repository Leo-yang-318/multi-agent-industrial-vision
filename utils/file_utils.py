from pathlib import Path


def ensure_file_exists(file_path: str) -> None:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")


def ensure_dir_exists(dir_path: str) -> None:
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)