from pathlib import Path
import random
import shutil


def copy_files(files, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        shutil.copy2(file_path, target_dir / file_path.name)


def split_stage4_dataset(
        source_root="data/stage4_anomaly/my_part",
        target_root="data/stage4_anomaly/my_part_split",
        seed=42,
):
    source_root = Path(source_root)
    target_root = Path(target_root)

    good_dir = source_root / "good"
    defect_dir = source_root / "defect"

    if not good_dir.exists():
        raise FileNotFoundError(f"正常样本目录不存在: {good_dir}")
    if not defect_dir.exists():
        raise FileNotFoundError(f"异常样本目录不存在: {defect_dir}")

    good_files = sorted(good_dir.glob("*.bmp"))
    defect_files = sorted(defect_dir.glob("*.bmp"))

    if len(good_files) < 230:
        raise ValueError(f"正常图数量不足，需要 230 张，当前只有 {len(good_files)} 张")
    if len(defect_files) < 70:
        raise ValueError(f"异常图数量不足，需要 70 张，当前只有 {len(defect_files)} 张")

    random.seed(seed)
    random.shuffle(good_files)
    random.shuffle(defect_files)

    train_good = good_files[:200]
    val_good = good_files[200:220]
    test_good = good_files[220:240]

    val_defect = defect_files[:30]
    test_defect = defect_files[30:50]

    # 如果目标目录已经存在，建议先删除，避免混入旧文件
    if target_root.exists():
        shutil.rmtree(target_root)

    copy_files(train_good, target_root / "train" / "good")
    copy_files(val_good, target_root / "val" / "good")
    copy_files(test_good, target_root / "test" / "good")

    copy_files(val_defect, target_root / "val" / "defect")
    copy_files(test_defect, target_root / "test" / "defect")

    print("数据集划分完成：")
    print(f"train/good: {len(train_good)}")
    print(f"val/good: {len(val_good)}")
    print(f"val/defect: {len(val_defect)}")
    print(f"test/good: {len(test_good)}")
    print(f"test/defect: {len(test_defect)}")
    print(f"输出目录: {target_root.resolve()}")


if __name__ == "__main__":
    split_stage4_dataset()
