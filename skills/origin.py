from pathlib import Path
import json

import cv2
import numpy as np


def make_line_kernel(length, angle_deg, thickness=3):
    size = max(3, int(length) | 1)
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    rad = np.deg2rad(angle_deg)
    dx = int(round(np.cos(rad) * (size // 2 - 1)))
    dy = int(round(np.sin(rad) * (size // 2 - 1)))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, thickness=thickness)
    return kernel


def contour_geometry(contour):
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (rect_w, rect_h), angle = rect
    long_side = max(rect_w, rect_h)
    short_side = max(1.0, min(rect_w, rect_h))
    angle = angle if rect_w >= rect_h else angle + 90.0
    return {
        "rect": rect,
        "center": (float(center_x), float(center_y)),
        "long_side": float(long_side),
        "short_side": float(short_side),
        "angle": float(angle),
        "aspect_ratio": float(long_side / short_side)
    }


def ensure_dir(path: Path):
    """确保输出文件夹存在"""
    path.mkdir(parents=True, exist_ok=True)


def read_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像，请检查路径是否正确：{image_path}")
    return image


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(2.0, (8, 8))  # 提高图像对比度，避免噪声过度增强
    enhanced = clahe.apply(blur)
    return gray, enhanced


def get_part_mask(gray):  # 获取主体工件避免误检背景
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio > 0.6:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 闭运算：填补工件内部小空洞
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 开运算：去掉小噪点
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


def get_filled_part_mask(part_mask):
    """
    生成填满孔洞的工件 mask。

    原来的 part_mask 中，孔洞可能是黑色。
    但检测孔时，我们需要知道孔也位于工件内部，
    所以要把工件外轮廓整体填满。
    """
    contours, _ = cv2.findContours(
        part_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(part_mask, dtype=np.uint8)

    if not contours:
        return filled_mask

    main_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(
        filled_mask,
        [main_contour],
        -1,
        255,
        thickness=-1
    )

    return filled_mask


def get_hole_ignore_mask(gray, part_mask):
    """
    生成孔周围忽略区域 mask。
    正常输出应该是：孔及孔周围一圈为白色，其余区域为黑色。
    """
    filled_part_mask = get_filled_part_mask(part_mask)

    # 只在“填满孔洞后的工件区域”内找黑色孔
    masked_gray = cv2.bitwise_and(gray, gray, mask=filled_part_mask)

    # 孔一般是黑色，所以用低灰度阈值提取
    _, dark_region = cv2.threshold(
        masked_gray,
        80,
        255,
        cv2.THRESH_BINARY_INV
    )

    # 只保留工件内部的黑色区域，避免背景被检测成孔
    dark_region = cv2.bitwise_and(
        dark_region,
        dark_region,
        mask=filled_part_mask
    )

    # 去掉真正背景区域
    # 因为背景也是黑的，必须再次限制在工件外轮廓内部
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_region = cv2.morphologyEx(
        dark_region,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        dark_region,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    hole_mask = np.zeros_like(gray, dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 100:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(1, min(w, h))

        # 孔：面积合适、接近圆形
        if circularity > 0.45 and aspect_ratio < 1.8:
            cv2.drawContours(
                hole_mask,
                [contour],
                -1,
                255,
                thickness=-1
            )

    # 孔周围扩大一圈，把孔边缘高亮圆弧也排除
    ignore_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (45, 45)
    )
    hole_ignore_mask = cv2.dilate(
        hole_mask,
        ignore_kernel,
        iterations=1
    )

    return hole_ignore_mask


def get_edge_ignore_mask(part_mask, edge_width=35):
    """
    生成工件边缘忽略区域 mask。

    目的：
    工件外边缘经常有高亮反光，容易被误判成划痕。
    所以把工件边缘向内一定宽度的区域排除掉。

    参数：
    part_mask: 工件主体 mask，工件为白色，背景为黑色
    edge_width: 忽略边缘宽度，数值越大，排除范围越宽
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (edge_width, edge_width)
    )

    eroded_mask = cv2.erode(
        part_mask,
        kernel,
        iterations=1
    )

    edge_ignore_mask = cv2.subtract(
        part_mask,
        eroded_mask
    )

    return edge_ignore_mask


def grow_scratch_bbox(seed_contour, response, part_mask):
    """
    沿着种子划痕方向扩展检测框。

    改进点：
    不再只找与种子连通的区域，而是在划痕方向上建立长条搜索带，
    把搜索带内的高响应点整体用于生成旋转框。
    这样可以覆盖断裂的长划痕。
    """
    geometry = contour_geometry(seed_contour)
    center_x, center_y = geometry["center"]
    seed_long = geometry["long_side"]
    seed_short = geometry["short_side"]
    seed_angle = geometry["angle"]

    x, y, w, h = cv2.boundingRect(seed_contour)

    # 搜索范围放大，防止只看局部
    pad = int(max(500, seed_long * 10))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(response.shape[1], x + w + pad)
    y1 = min(response.shape[0], y + h + pad)

    roi = response[y0:y1, x0:x1]
    roi_mask = part_mask[y0:y1, x0:x1]

    ys, xs = np.indices(roi.shape)
    xs_global = xs + x0
    ys_global = ys + y0

    rad = np.deg2rad(seed_angle)
    unit_x = np.cos(rad)
    unit_y = np.sin(rad)

    # 点到划痕方向直线的投影距离
    projection = (xs_global - center_x) * unit_x + (ys_global - center_y) * unit_y

    # 点到划痕中心线的垂直距离
    perpendicular = np.abs(
        -(xs_global - center_x) * unit_y + (ys_global - center_y) * unit_x
    )

    # 搜索带宽度，太小会漏，太大会误检
    band_half_width = max(10, int(seed_short * 2))

    # 沿划痕方向向两侧搜索得更远
    extension = max(500, int(seed_long * 10))

    band_mask = (
            (perpendicular <= band_half_width) &
            (projection >= -extension) &
            (projection <= extension) &
            (roi_mask > 0)
    )

    band_values = roi[band_mask]

    if band_values.size < 50:
        return cv2.boundingRect(seed_contour), geometry

    # 阈值不要太高，否则后半段弱划痕会断掉
    threshold = float(np.percentile(band_values, 65))

    candidate_mask = (
                             (roi >= threshold) &
                             band_mask
                     ).astype(np.uint8) * 255

    # 用方向核把同方向的断裂划痕连接起来
    line_kernel = make_line_kernel(
        length=121,
        angle_deg=seed_angle,
        thickness=max(5, int(seed_short * 2))
    )

    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        line_kernel,
        iterations=1
    )

    # 去掉小噪点
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    # 注意：这里不再只取与 seed 连通的 component
    # 而是取搜索带内所有候选点
    point_y, point_x = np.where(candidate_mask > 0)

    if len(point_x) < 20:
        return cv2.boundingRect(seed_contour), geometry

    points = np.column_stack((point_x + x0, point_y + y0)).astype(np.int32)
    points = points.reshape(-1, 1, 2)

    grown_geometry = contour_geometry(points)
    grown_bbox = cv2.boundingRect(points)

    return grown_bbox, grown_geometry


def is_same_scratch(candidate, existing):
    candidate_bbox = candidate["bbox"]
    existing_bbox = existing["bbox"]

    x1 = max(candidate_bbox[0], existing_bbox[0])
    y1 = max(candidate_bbox[1], existing_bbox[1])
    x2 = min(
        candidate_bbox[0] + candidate_bbox[2],
        existing_bbox[0] + existing_bbox[2]
    )
    y2 = min(
        candidate_bbox[1] + candidate_bbox[3],
        existing_bbox[1] + existing_bbox[3]
    )
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    candidate_area = candidate_bbox[2] * candidate_bbox[3]
    existing_area = existing_bbox[2] * existing_bbox[3]
    union = candidate_area + existing_area - intersection
    iou = intersection / union if union > 0 else 0.0

    candidate_center_y = candidate_bbox[1] + candidate_bbox[3] / 2.0
    existing_center_y = existing_bbox[1] + existing_bbox[3] / 2.0

    return (
            iou > 0.15 or
            (
                    abs(candidate["angle"] - existing["angle"]) < 10.0 and
                    abs(candidate_center_y - existing_center_y) < 220.0
            )
    )


def detect_scratch(
        enhanced,
        part_mask,
        hole_ignore_mask=None,
        edge_ignore_mask=None
):
    defects = []
    # 同时保留暗划痕和亮划痕响应，避免只命中划痕局部的单侧边缘。
    response_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, response_kernel)
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, response_kernel)
    response = cv2.max(blackhat, tophat)

    # 只在工件内部统计响应分位数，避免全局阈值被背景带偏。
    response_values = response[part_mask > 0]
    threshold = float(np.percentile(response_values, 96))
    _, scratch_binary = cv2.threshold(
        response,
        threshold,
        255,
        cv2.THRESH_BINARY
    )
    scratch_binary = cv2.bitwise_and(scratch_binary, scratch_binary, mask=part_mask)

    # 去掉孔周围区域，避免孔边缘圆弧被误判为划痕
    if hole_ignore_mask is not None:
        scratch_binary[hole_ignore_mask > 0] = 0

    if edge_ignore_mask is not None:
        scratch_binary[edge_ignore_mask > 0] = 0

    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    scratch_binary = cv2.morphologyEx(
        scratch_binary,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        scratch_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        # 面积太小，认为是噪声
        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        seed_geometry = contour_geometry(contour)
        long_side = seed_geometry["long_side"]
        aspect_ratio = seed_geometry["aspect_ratio"]

        # 先放宽种子条件，把另外两条较弱的划痕也纳入候选。
        if aspect_ratio >= 3 and long_side >= 25:
            grown_bbox, grown_geometry = grow_scratch_bbox(
                contour,
                response,
                part_mask
            )
            gx, gy, gw, gh = grown_bbox
            grow_factor = (gw * gh) / max(1.0, w * h)

            box = cv2.boxPoints(grown_geometry["rect"])
            box = np.int32(box)

            # 扩展后的结构仍需保持细长，同时限制过度蔓延的假阳性。
            if (
                    grown_geometry["aspect_ratio"] < 2.5 or
                    grown_geometry["long_side"] < 80 or
                    grow_factor > 500
            ):
                continue

            base_score = long_side * aspect_ratio
            grow_penalty = 1.0 + max(0.0, grow_factor - 15.0) / 15.0
            area_penalty = (gw * gh) / 1000.0
            score = base_score / (grow_penalty * area_penalty)

            defects.append({
                "type": "scratch",
                "name": "划痕",
                "bbox": [int(gx), int(gy), int(gw), int(gh)],
                "rotated_box": box.tolist(),
                "area": float(area),
                "seed_bbox": [int(x), int(y), int(w), int(h)],
                "long_side": float(grown_geometry["long_side"]),
                "short_side": float(grown_geometry["short_side"]),
                "aspect_ratio": float(grown_geometry["aspect_ratio"]),
                "angle": float(grown_geometry["angle"]),
                "grow_factor": float(grow_factor),
                "score": float(score),
                "rule": "response p96 seed inside part_mask, seed aspect_ratio >= 3 and seed long_side >= 25, then grow bbox and keep elongated components"

            })

    unique_defects = []
    for defect in sorted(defects, key=lambda item: item["score"], reverse=True):
        if any(is_same_scratch(defect, existing) for existing in unique_defects):
            continue
        unique_defects.append(defect)

    return unique_defects, blackhat, scratch_binary


def draw_results(image, defects):
    """
    在原图上绘制检测结果。
    如果有 rotated_box，则画旋转框；
    如果没有 rotated_box，则画普通水平框。
    """
    vis = image.copy()

    for defect in defects:
        name = defect["type"]

        if "rotated_box" in defect:
            box = np.array(defect["rotated_box"], dtype=np.int32)
            cv2.polylines(
                vis,
                [box],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2
            )

            text_x = int(box[:, 0].min())
            text_y = int(box[:, 1].min())

            cv2.putText(
                vis,
                name,
                (text_x, max(20, text_y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        else:
            x, y, w, h = defect["bbox"]

            cv2.rectangle(
                vis,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                vis,
                name,
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    return vis


def run_scratch_detection(
        image_path: str,
        output_dir: str = "outputs/traditional_defect"
):
    """
    划痕检测主函数。

    输入：
    image_path: 图片路径
    output_dir: 输出目录

    输出：
    result.json
    result_vis.jpg
    part_mask.jpg
    enhanced.jpg
    scratch_blackhat.jpg
    scratch_binary.jpg
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    image = read_image(str(image_path))

    gray, enhanced = preprocess(image)

    part_mask = get_part_mask(gray)

    hole_ignore_mask = get_hole_ignore_mask(gray, part_mask)

    edge_ignore_mask = get_edge_ignore_mask(part_mask, edge_width=45)

    scratch_defects, blackhat, scratch_binary = detect_scratch(
        enhanced,
        part_mask,
        hole_ignore_mask,
        edge_ignore_mask
    )

    result = {
        "image_path": str(image_path),
        "task": "scratch_detection",
        "has_defect": len(scratch_defects) > 0,
        "defect_count": len(scratch_defects),
        "defects": scratch_defects
    }

    # 保存 JSON 结果
    result_json_path = output_dir / "result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存可视化检测结果
    vis = draw_results(image, scratch_defects)
    cv2.imwrite(str(output_dir / "result_vis.jpg"), vis)

    # 保存中间调试结果
    cv2.imwrite(str(output_dir / "gray.jpg"), gray)
    cv2.imwrite(str(output_dir / "enhanced.jpg"), enhanced)
    cv2.imwrite(str(output_dir / "part_mask.jpg"), part_mask)
    cv2.imwrite(str(output_dir / "hole_ignore_mask.jpg"), hole_ignore_mask)
    cv2.imwrite(str(output_dir / "edge_ignore_mask.jpg"), edge_ignore_mask)
    cv2.imwrite(str(output_dir / "scratch_blackhat.jpg"), blackhat)
    cv2.imwrite(str(output_dir / "scratch_binary.jpg"), scratch_binary)

    return result


if __name__ == "__main__":
    image_path = r"D:\project\data\input_images\scratch\Image_20260130101410030.bmp"

    result = run_scratch_detection(
        image_path=image_path,
        output_dir=r"D:\project\outputs\traditional_defect"
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
