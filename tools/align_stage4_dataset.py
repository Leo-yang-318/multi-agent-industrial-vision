from pathlib import Path
import shutil

import cv2
import numpy as np

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png"}


def segment_workpiece(image: np.ndarray) -> np.ndarray:
    """
    分割浅色工件区域。
    输出 mask：工件区域为 255，背景为 0。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    clean_mask = np.zeros_like(mask)

    if not contours:
        return clean_mask

    max_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(max_contour) < 100:
        return clean_mask

    cv2.drawContours(clean_mask, [max_contour], -1, 255, thickness=-1)

    return clean_mask


def estimate_angle_by_min_area_rect(mask: np.ndarray) -> float:
    """
    通过工件最大轮廓的最小外接矩形估计粗略旋转角度。
    注意：你的工件接近正方形，所以这一步只做粗矫正。
    后面还会用孔洞分布做 0/90/180/270 方向统一。
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)

    (_, _), (w, h), angle = rect

    if w < h:
        rotate_angle = angle
    else:
        rotate_angle = angle + 90

    return rotate_angle


def rotate_keep_all(image: np.ndarray, mask: np.ndarray, angle: float):
    """
    旋转图像和 mask，同时扩大画布，避免工件被裁掉。
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_image, rotated_mask


def crop_and_center(
        image: np.ndarray,
        mask: np.ndarray,
        output_size: int = 512,
        margin_ratio: float = 0.15,
) -> np.ndarray:
    """
    根据工件 mask 裁剪 ROI，并将工件居中、统一尺寸。
    最终输出固定大小图像，例如 512×512。
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return cv2.resize(image, (output_size, output_size))

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1

    margin = int(max(box_w, box_h) * margin_ratio)

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.shape[1] - 1, x2 + margin)
    y2 = min(image.shape[0] - 1, y2 + margin)

    crop_img = image[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]

    # 背景统一填黑，只保留工件区域。
    crop_img = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)

    h, w = crop_img.shape[:2]
    canvas_size = max(h, w)

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    y_offset = (canvas_size - h) // 2
    x_offset = (canvas_size - w) // 2

    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = crop_img

    resized = cv2.resize(
        canvas,
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def extract_hole_mask(aligned_image: np.ndarray) -> np.ndarray:
    """
    提取工件内部黑色孔洞区域。
    这里先将图像灰度化、二值化，让孔洞更明显。
    输出 mask：孔洞为 255，其余为 0。
    """
    gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 提取亮色工件主体。
    _, workpiece_mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((7, 7), np.uint8)
    workpiece_mask = cv2.morphologyEx(
        workpiece_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        workpiece_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    clean_workpiece = np.zeros_like(workpiece_mask)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_workpiece, [max_contour], -1, 255, thickness=-1)

    workpiece_mask = clean_workpiece

    # 提取黑色孔洞候选区域。
    # 如果孔洞提取不全，可以把 90 调大到 100 或 110。
    # 如果污点、阴影也被当成孔洞，可以把 90 调小到 70 或 80。
    hole_candidate = cv2.inRange(gray, 0, 90)

    # 腐蚀工件区域，尽量去掉边缘背景干扰。
    inner_workpiece = cv2.erode(
        workpiece_mask,
        np.ones((9, 9), np.uint8),
        iterations=1,
    )

    hole_mask = cv2.bitwise_and(hole_candidate, inner_workpiece)

    # 开操作去掉小噪声。
    hole_mask = cv2.morphologyEx(
        hole_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hole_mask,
        connectivity=8,
    )

    components = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        aspect_ratio = w / (h + 1e-8)

        # 过滤太小噪点、太大黑块、过细长划痕。
        if 50 <= area <= 8000 and 0.3 <= aspect_ratio <= 3.0:
            components.append((area, label_id))

    # 只保留较大的若干孔洞区域，降低缺陷黑点干扰。
    # 如果你的工件孔洞数量很多，可以把 20 调大。
    components = sorted(components, reverse=True)[:20]

    clean_holes = np.zeros_like(hole_mask)

    for _, label_id in components:
        clean_holes[labels == label_id] = 255

    return clean_holes


def rotate_by_90(image: np.ndarray, k: int) -> np.ndarray:
    """
    k=0: 不旋转
    k=1: 顺时针 90°
    k=2: 旋转 180°
    k=3: 逆时针 90°
    """
    if k == 0:
        return image
    if k == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    if k == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def choose_best_orientation_by_holes(
        aligned_image: np.ndarray,
        reference_hole_mask: np.ndarray,
):
    """
    枚举 0/90/180/270 四个方向，
    选择孔洞布局与参考图最相似的方向。
    返回：最佳图像、最佳旋转编号、相似度分数。
    """
    best_score = -1.0
    best_image = aligned_image
    best_k = 0

    for k in range(4):
        candidate = rotate_by_90(aligned_image, k)
        candidate_hole_mask = extract_hole_mask(candidate)

        candidate_hole_mask = cv2.resize(
            candidate_hole_mask,
            (reference_hole_mask.shape[1], reference_hole_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        intersection = np.logical_and(
            candidate_hole_mask > 0,
            reference_hole_mask > 0,
        ).sum()

        union = np.logical_or(
            candidate_hole_mask > 0,
            reference_hole_mask > 0,
        ).sum()

        score = intersection / (union + 1e-8)

        if score > best_score:
            best_score = score
            best_image = candidate
            best_k = k

    return best_image, best_k, best_score


def preprocess_one_image(
        image: np.ndarray,
        output_size: int = 512,
) -> np.ndarray | None:
    """
    单张图像预处理：
    分割工件 → 外轮廓粗角度矫正 → 裁剪 ROI → 工件居中 → 统一尺寸。
    """
    mask = segment_workpiece(image)

    if np.count_nonzero(mask) == 0:
        return None

    angle = estimate_angle_by_min_area_rect(mask)

    rotated_image, rotated_mask = rotate_keep_all(
        image=image,
        mask=mask,
        angle=angle,
    )

    aligned = crop_and_center(
        image=rotated_image,
        mask=rotated_mask,
        output_size=output_size,
        margin_ratio=0.15,
    )

    return aligned


def align_one_image(
        src_path: Path,
        dst_path: Path,
        reference_hole_mask: np.ndarray,
        output_size: int = 512,
        save_debug: bool = True,
):
    """
    处理单张图片，并根据参考孔洞布局统一方向。
    """
    image = cv2.imread(str(src_path))

    if image is None:
        print(f"读取失败，跳过: {src_path}")
        return

    aligned = preprocess_one_image(
        image=image,
        output_size=output_size,
    )

    if aligned is None:
        print(f"工件分割失败，直接复制: {src_path.name}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return

    aligned, best_k, best_score = choose_best_orientation_by_holes(
        aligned_image=aligned,
        reference_hole_mask=reference_hole_mask,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), aligned)

    if save_debug:
        debug_dir = dst_path.parent / "_debug_holes"
        debug_dir.mkdir(parents=True, exist_ok=True)

        hole_mask = extract_hole_mask(aligned)

        cv2.imwrite(
            str(debug_dir / f"{src_path.stem}_hole_mask.png"),
            hole_mask,
        )

    print(
        f"aligned: {src_path.name}, "
        f"rotate90_k={best_k}, "
        f"hole_score={best_score:.4f}"
    )


def build_reference_hole_mask(
        image_paths: list[Path],
        output_size: int = 512,
        reference_image_path: str | None = None,
) -> np.ndarray:
    """
    生成参考孔洞 mask。
    如果传入 reference_image_path，就使用手动指定的参考图。
    否则自动从 train/good 中选第一张可用图。
    """
    if reference_image_path is not None:
        reference_path = Path(reference_image_path)

        if not reference_path.exists():
            raise FileNotFoundError(f"手动指定的参考图不存在: {reference_path}")

        reference_candidates = [reference_path]
    else:
        reference_candidates = [
            p for p in image_paths
            if "train" in p.parts and "good" in p.parts
        ]

        reference_candidates = sorted(reference_candidates)

    if not reference_candidates:
        raise ValueError("没有找到可用于参考方向的图片")

    for reference_path in reference_candidates:
        image = cv2.imread(str(reference_path))

        if image is None:
            continue

        aligned = preprocess_one_image(
            image=image,
            output_size=output_size,
        )

        if aligned is None:
            continue

        reference_hole_mask = extract_hole_mask(aligned)

        if np.count_nonzero(reference_hole_mask) == 0:
            continue

        print(f"reference image = {reference_path}")
        return reference_hole_mask

    raise ValueError("没有找到可用于提取孔洞参考方向的图片")


def align_dataset(
        source_root: str = "data/stage4_anomaly/my_part_split",
        target_root: str = "data/stage4_anomaly/my_part_split_aligned_roi",
        output_size: int = 512,
        reference_image_path: str | None = None,
        save_debug: bool = True,
):
    """
    对整个数据集进行：
    方向矫正、ROI 裁剪、工件居中、统一尺寸、孔洞方向统一。
    """
    source_root = Path(source_root)
    target_root = Path(target_root)

    if not source_root.exists():
        raise FileNotFoundError(f"源数据集不存在: {source_root}")

    if target_root.exists():
        shutil.rmtree(target_root)

    image_paths = [
        p for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not image_paths:
        raise ValueError(f"没有找到图片: {source_root}")

    print(f"source_root = {source_root.resolve()}")
    print(f"target_root = {target_root.resolve()}")
    print(f"total images = {len(image_paths)}")

    reference_hole_mask = build_reference_hole_mask(
        image_paths=image_paths,
        output_size=output_size,
        reference_image_path=reference_image_path,
    )

    for src_path in image_paths:
        relative_path = src_path.relative_to(source_root)
        dst_path = target_root / relative_path

        align_one_image(
            src_path=src_path,
            dst_path=dst_path,
            reference_hole_mask=reference_hole_mask,
            output_size=output_size,
            save_debug=save_debug,
        )

    print("方向矫正 + ROI 裁剪 + 工件居中 + 孔洞方向统一完成。")


if __name__ == "__main__":
    align_dataset(
        source_root="data/stage4_anomaly/my_part_split",
        target_root="data/stage4_anomaly/my_part_split_aligned_roi",
        output_size=512,
        reference_image_path=None,
        save_debug=False,
    )
