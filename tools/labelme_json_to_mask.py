from pathlib import Path
import json

import cv2
import numpy as np


def labelme_json_to_mask(json_path: Path, save_dir: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_height = data["imageHeight"]
    image_width = data["imageWidth"]

    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for shape in data.get("shapes", []):
        label = shape.get("label", "")

        if label != "defect":
            continue

        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    save_dir.mkdir(parents=True, exist_ok=True)

    mask_name = json_path.stem + "_mask.png"
    save_path = save_dir / mask_name

    cv2.imwrite(str(save_path), mask)

    print(f"saved mask: {save_path}")


def convert_all_labelme_jsons(
        json_dir: str = r"D:\project\data\stage4_anomaly\my_part_split\test\defect",
        mask_dir: str = r"D:\project\data\stage4_anomaly\my_part_split\ground_truth\defect",
):
    json_dir = Path(json_dir)
    mask_dir = Path(mask_dir)

    if not json_dir.exists():
        raise FileNotFoundError(f"json 目录不存在: {json_dir}")

    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        raise ValueError(f"没有找到 json 文件: {json_dir}")

    print(f"json_dir = {json_dir.resolve()}")
    print(f"mask_dir = {mask_dir.resolve()}")
    print(f"json 数量 = {len(json_files)}")

    for json_path in json_files:
        labelme_json_to_mask(json_path, mask_dir)

    print("全部 json 已转换为 mask。")


if __name__ == "__main__":
    convert_all_labelme_jsons()
