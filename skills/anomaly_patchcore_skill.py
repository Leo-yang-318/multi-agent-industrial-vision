from pathlib import Path
import csv
import pandas as pd
import cv2
import numpy as np

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

from scipy.stats import spearmanr

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png"}


def _count_files(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def load_defect_labels(csv_path: Path) -> dict:
    """
    读取缺陷类别标注文件。

    返回格式：
    {
        "Image_xxx.bmp": {
            "defect_type": "scratch",
            "defect_shape": "line"
        }
    }
    """

    defect_labels = {}

    if not csv_path.exists():
        print(f"未找到缺陷类别文件: {csv_path}")
        return defect_labels

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_name = row["image_name"].strip()
            defect_type = row["defect_type"].strip()
            defect_shape = row["defect_shape"].strip()

            defect_labels[image_name] = {
                "defect_type": defect_type,
                "defect_shape": defect_shape,
            }

    print(f"已读取缺陷类别数量: {len(defect_labels)}")

    return defect_labels


def calculate_heatmap_gradient_score(
        anomaly_map,
        gt_mask
):
    h, w = gt_mask.shape

    anomaly_map = cv2.resize(
        anomaly_map,
        (w, h)
    )

    # 归一化
    anomaly_map = (
                          anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    # 缺陷区域热力

    defect_score = anomaly_map[
        gt_mask > 0
        ].mean()

    # 背景热力

    background_score = anomaly_map[
        gt_mask == 0
        ].mean()

    # 差值

    score_gap = (
            defect_score -
            background_score
    )

    # ==================
    # 距离梯度
    # ==================

    distance_map = cv2.distanceTransform(
        gt_mask.astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    distance_map = distance_map.max() - distance_map

    corr, _ = spearmanr(
        distance_map.flatten(),
        anomaly_map.flatten()
    )

    return {

        "Defect_Mean_Score":
            float(defect_score),

        "Background_Mean_Score":
            float(background_score),

        "Score_Gap":
            float(score_gap),

        "Gradient_Correlation":
            float(corr)

    }


def run_patchcore(
        dataset_root: str = "data/stage4_anomaly/my_part_split",
        output_dir: str = "outputs/stage4_anomaly",
        category_name: str = "my_part_split",
        train_normal_dir: str = "train/good",
        test_normal_dir: str = "test/good",
        test_abnormal_dir: str = "test/defect",
        test_split_ratio: float = 0.2,
        val_split_mode: str = "from_test",
        val_split_ratio: float = 0.5,
        seed: int = 42,
        pixel_threshold: float = 0.75,
        min_area: int = 250,
        defect_label_csv: str = "data/stage4_anomaly/my_part_split/defect_labels.csv",
) -> dict:
    dataset_root_path = Path(dataset_root)
    output_dir_path = Path(output_dir)
    defect_label_path = Path(defect_label_csv)
    defect_labels = load_defect_labels(defect_label_path)
    train_normal_dir_path = dataset_root_path / train_normal_dir
    test_normal_dir_path = dataset_root_path / test_normal_dir
    test_abnormal_dir_path = dataset_root_path / test_abnormal_dir

    if not dataset_root_path.exists():
        raise FileNotFoundError(f"数据集根目录不存在: {dataset_root_path}")

    if not train_normal_dir_path.exists():
        raise FileNotFoundError(f"训练正常样本目录不存在: {train_normal_dir_path}")

    if not test_normal_dir_path.exists():
        raise FileNotFoundError(f"测试正常样本目录不存在: {test_normal_dir_path}")

    if not test_abnormal_dir_path.exists():
        raise FileNotFoundError(f"测试异常样本目录不存在: {test_abnormal_dir_path}")

    train_normal_count = _count_files(train_normal_dir_path)
    test_normal_count = _count_files(test_normal_dir_path)
    test_abnormal_count = _count_files(test_abnormal_dir_path)

    if train_normal_count == 0:
        raise ValueError(f"训练正常样本目录为空: {train_normal_dir_path}")

    if test_normal_count == 0:
        raise ValueError(f"测试正常样本目录为空: {test_normal_dir_path}")

    if test_abnormal_count == 0:
        raise ValueError(f"测试异常样本目录为空: {test_abnormal_dir_path}")

    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"dataset_root = {dataset_root_path.resolve()}")
    print(f"output_dir = {output_dir_path.resolve()}")
    print(f"train_normal_dir = {train_normal_dir_path.resolve()} ({train_normal_count} files)")
    print(f"test_normal_dir = {test_normal_dir_path.resolve()} ({test_normal_count} files)")
    print(f"test_abnormal_dir = {test_abnormal_dir_path.resolve()} ({test_abnormal_count} files)")
    print(
        "split_config = "
        f"test_split_mode=from_dir, test_split_ratio={test_split_ratio}, "
        f"val_split_mode={val_split_mode}, val_split_ratio={val_split_ratio}, seed={seed}"
    )

    datamodule = Folder(

        name=category_name,

        root=dataset_root_path,

        normal_dir=train_normal_dir,

        normal_test_dir=test_normal_dir,

        abnormal_dir=test_abnormal_dir,

        extensions=[
            ".bmp",
            ".png",
            ".jpg",
            ".jpeg"
        ],

        test_split_mode="from_dir",
        num_workers=0,
        seed=seed
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    engine = Engine(
        default_root_dir=output_dir_path,
    )

    datamodule.setup()
    train_count = len(datamodule.train_data)
    test_count = len(datamodule.test_data)
    val_count = len(datamodule.val_data) if hasattr(datamodule, "val_data") else 0

    print(
        f"dataset_split = train:{train_count}, val:{val_count}, test:{test_count}"
    )
    print("开始训练 PatchCore...")

    engine.fit(
        model=model,
        datamodule=datamodule
    )

    print("开始测试 PatchCore...")
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("开始预测并进行自定义阈值后处理...")

    predictions = list(
        engine.predict(
            model=model,
            datamodule=datamodule,
        )
    )
    image_results = []
    # =========================
    # 生成预测 mask 并计算像素级指标
    # =========================

    # 设置预测 mask 保存目录
    custom_mask_dir = output_dir_path / "custom_threshold_masks"
    custom_mask_dir.mkdir(parents=True, exist_ok=True)

    # 人工 mask 目录
    gt_mask_dir = dataset_root_path / "ground_truth" / "defect"  # 你人工标注 mask 的路径

    pixel_results = []
    heatmap_results = []

    for batch in predictions:
        image_paths = batch.image_path
        anomaly_maps = batch.anomaly_map

        for image_path, anomaly_map in zip(image_paths, anomaly_maps):
            image_path = Path(image_path)
            image_name = image_path.name

            label_info = defect_labels.get(
                image_name,
                {
                    "defect_type": "normal_or_unknown",
                    "defect_shape": "unknown",
                },
            )

            defect_type = label_info["defect_type"]
            defect_shape = label_info["defect_shape"]

            single_result = {
                "image_name": image_name,
                "defect_type": defect_type,
                "defect_shape": defect_shape,

                "defect_mean_score": None,
                "background_mean_score": None,
                "score_gap": None,
                "gradient_correlation": None,

                "pixel_accuracy": None,
                "pixel_precision": None,
                "pixel_recall": None,
                "iou": None,
                "dice": None,
            }

            print(f"当前图片: {image_name}")
            print(f"缺陷类型: {defect_type}, 缺陷形态: {defect_shape}")

            # 归一化 anomaly_map
            amap = anomaly_map.detach().cpu().numpy()
            if amap.ndim == 3:
                amap = np.squeeze(amap)
            amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

            # 阈值二值化
            mask = (amap_norm > pixel_threshold).astype(np.uint8) * 255

            # 连通域过滤，去掉小区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            filtered_mask = np.zeros_like(mask)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= min_area:
                    filtered_mask[labels == label_id] = 255

            # 保存预测 mask
            save_path = custom_mask_dir / f"{image_path.stem}_custom_mask.png"
            cv2.imwrite(str(save_path), filtered_mask)

            # ==========================
            # 对比人工mask
            # ==========================

            gt_path = (
                    gt_mask_dir /
                    f"{image_path.stem}_mask.png"
            )
            print(
                "寻找mask:",
                gt_path
            )

            if gt_path.exists():
                gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

                if gt_mask is not None:
                    gt_binary = gt_mask > 127

                    # 计算热力空间指标
                    heat_result = calculate_heatmap_gradient_score(amap, gt_binary)
                    heatmap_results.append(heat_result)

                    # 把当前图片的热力指标保存到 single_result 里
                    single_result["defect_mean_score"] = heat_result["Defect_Mean_Score"]
                    single_result["background_mean_score"] = heat_result["Background_Mean_Score"]
                    single_result["score_gap"] = heat_result["Score_Gap"]
                    single_result["gradient_correlation"] = heat_result["Gradient_Correlation"]

                    # 后面开始计算像素级指标
                    pred_binary = filtered_mask > 127

                # ==========================
                # 新增：热力空间评价
                # ==========================

                heat_result = calculate_heatmap_gradient_score(
                    amap,
                    gt_binary
                )

                heatmap_results.append(
                    heat_result
                )

                # ==========================
                # 原来的像素mask评价
                # 可以保留
                # ==========================

                pred_binary = (
                        filtered_mask > 127
                ).astype(np.uint8)

                # 后面原来的：
                # IoU
                # Dice
                # Pixel Accuracy

                # 继续保留

                # 如果预测 mask 和人工 mask 尺寸不一致，把预测 mask 调整到人工 mask 的尺寸
                if pred_binary.shape != gt_binary.shape:
                    gt_h, gt_w = gt_binary.shape

                    pred_binary = cv2.resize(
                        pred_binary,
                        (gt_w, gt_h),  # 注意：cv2.resize 的顺序是 宽, 高
                        interpolation=cv2.INTER_NEAREST,
                    )

                # 转成 bool，方便做逻辑运算
                pred_bool = pred_binary.astype(bool)
                gt_bool = gt_binary.astype(bool)

                pixel_tp = np.logical_and(pred_bool, gt_bool).sum()
                pixel_tn = np.logical_and(~pred_bool, ~gt_bool).sum()
                pixel_fp = np.logical_and(pred_bool, ~gt_bool).sum()
                pixel_fn = np.logical_and(~pred_bool, gt_bool).sum()

                pixel_accuracy = (pixel_tp + pixel_tn) / (pixel_tp + pixel_tn + pixel_fp + pixel_fn)
                pixel_precision = pixel_tp / (pixel_tp + pixel_fp) if (pixel_tp + pixel_fp) > 0 else 0
                pixel_recall = pixel_tp / (pixel_tp + pixel_fn) if (pixel_tp + pixel_fn) > 0 else 0
                iou = pixel_tp / (pixel_tp + pixel_fp + pixel_fn) if (pixel_tp + pixel_fp + pixel_fn) > 0 else 0
                dice = (2 * pixel_tp) / (2 * pixel_tp + pixel_fp + pixel_fn) if (
                                                                                        2 * pixel_tp + pixel_fp + pixel_fn) > 0 else 0

                pixel_results.append({
                    "image": image_path.name,
                    "iou": float(iou),
                    "dice": float(dice),
                    "pixel_accuracy": float(pixel_accuracy),
                    "pixel_precision": float(pixel_precision),
                    "pixel_recall": float(pixel_recall),
                })

                single_result["pixel_accuracy"] = pixel_accuracy
                single_result["pixel_precision"] = pixel_precision
                single_result["pixel_recall"] = pixel_recall
                single_result["iou"] = iou
                single_result["dice"] = dice

            image_results.append(single_result)
            result_path = output_dir_path / "evaluation_results.csv"

            df = pd.DataFrame(image_results)

            df.to_csv(
                result_path,
                index=False,
                encoding="utf-8-sig"
            )

            print(
                f"评价结果已经保存:{result_path}"
            )

    # 输出平均像素级指标
    if pixel_results:
        avg_iou = np.mean([r["iou"] for r in pixel_results])
        avg_dice = np.mean([r["dice"] for r in pixel_results])
        avg_precision = np.mean([r["pixel_precision"] for r in pixel_results])
        avg_recall = np.mean([r["pixel_recall"] for r in pixel_results])
        avg_accuracy = np.mean([r["pixel_accuracy"] for r in pixel_results])

        print("\n=== 像素级平均指标 ===")
        print(f"Pixel Accuracy: {avg_accuracy:.4f}")
        print(f"Pixel Precision: {avg_precision:.4f}")
        print(f"Pixel Recall: {avg_recall:.4f}")
        print(f"IoU: {avg_iou:.4f}")
        print(f"Dice: {avg_dice:.4f}")

    # ==================================
    # 热力空间评价
    # ==================================

    if heatmap_results:
        avg_defect_score = np.mean(
            [
                x["Defect_Mean_Score"]
                for x in heatmap_results
            ]
        )

        avg_background_score = np.mean(
            [
                x["Background_Mean_Score"]
                for x in heatmap_results
            ]
        )

        avg_score_gap = np.mean(
            [
                x["Score_Gap"]
                for x in heatmap_results
            ]
        )

        avg_gradient_corr = np.mean(
            [
                x["Gradient_Correlation"]
                for x in heatmap_results
            ]
        )

        print("\n====== 热力空间评价 ======")

        print(
            f"Defect Mean Score: {avg_defect_score:.4f}"
        )

        print(
            f"Background Mean Score: {avg_background_score:.4f}"
        )

        print(
            f"Score Gap: {avg_score_gap:.4f}"
        )

        print(
            f"Gradient Correlation: {avg_gradient_corr:.4f}"
        )

    tp = 0  # 异常图被正确判断为异常
    tn = 0  # 正常图被正确判断为正常
    fp = 0  # 正常图被误判为异常，也叫误报
    fn = 0  # 异常图被误判为正常，也叫漏检

    for batch in predictions:
        true_labels = batch.gt_label
        pred_labels = batch.pred_label

        for true_label, pred_label in zip(true_labels, pred_labels):
            true_label = int(true_label)
            pred_label = int(pred_label)

            if true_label == 1 and pred_label == 1:
                tp += 1
            elif true_label == 0 and pred_label == 0:
                tn += 1
            elif true_label == 0 and pred_label == 1:
                fp += 1
            elif true_label == 1 and pred_label == 0:
                fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"TP 真阳性/异常检出数量 = {tp}")
    print(f"TN 真阴性/正常判断正确数量 = {tn}")
    print(f"FP 假阳性/误报数量 = {fp}")
    print(f"FN 假阴性/漏检数量 = {fn}")
    print(f"Accuracy 准确率 = {accuracy:.4f}")
    print(f"Precision 精确率 = {precision:.4f}")
    print(f"Recall 召回率/缺陷检出率 = {recall:.4f}")
    print(f"FPR 误报率 = {fpr:.4f}")
    print(f"FNR 漏检率 = {fnr:.4f}")

    metrics = test_results[0] if test_results else {}
    metrics["TP"] = tp
    metrics["TN"] = tn
    metrics["FP"] = fp
    metrics["FN"] = fn
    metrics["Accuracy"] = accuracy
    metrics["Precision"] = precision
    metrics["Recall"] = recall
    metrics["FPR"] = fpr
    metrics["FNR"] = fnr

    if heatmap_results:
        metrics["Defect_Mean_Score"] = avg_defect_score

        metrics["Background_Mean_Score"] = avg_background_score

        metrics["Score_Gap"] = avg_score_gap

        metrics["Gradient_Correlation"] = avg_gradient_corr

    return {
        "stage": "stage4_algorithm_validation",
        "model": "PatchCore",
        "library": "Anomalib",
        "task": "image-level anomaly detection",
        "input": {
            "dataset_root": str(dataset_root_path),
            "train_normal_dir": str(train_normal_dir_path),
            "test_normal_dir": str(test_normal_dir_path),
            "test_abnormal_dir": str(test_abnormal_dir_path),
        },
        "split_config": {
            "test_split_mode": "from_dir",
            "test_split_ratio": test_split_ratio,
            "val_split_mode": val_split_mode,
            "seed": seed,
        },
        "postprocess_config": {
            "pixel_threshold": pixel_threshold,
            "min_area": min_area,
            "custom_mask_dir": str(output_dir_path / "custom_threshold_masks"),
        },
        "dataset_stats": {
            "train_normal_count": train_normal_count,
            "test_normal_count": test_normal_count,
            "test_abnormal_count": test_abnormal_count,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
        },
        "metrics": metrics,
        "artifacts": {
            "output_dir": str(output_dir_path),
        },
    }


# ======================================
# Agent调用接口
# ======================================


class PatchCoreSkill:
    name = "patchcore_skill"

    def run(
            self,
            image_path=None,
            defect_type=None,
            params=None
    ):
        """
        PatchCore统一调用入口


        image_path:
            输入图片


        defect_type:
            缺陷类型


        params:
            参数配置

        """

        if params is None:
            params = {}

        # =========================
        # 获取参数
        # =========================

        dataset_root = params.get(

            "dataset_root",

            "data/stage4_anomaly/my_part_split"

        )

        output_dir = params.get(

            "output_dir",

            "outputs/stage4_anomaly"

        )

        category_name = params.get(

            "category_name",

            "my_part_split"

        )

        pixel_threshold = params.get(

            "pixel_threshold",

            0.75

        )

        min_area = params.get(

            "min_area",

            250

        )

        # =========================
        # 调用原来的PatchCore
        # =========================

        result = run_patchcore(

            dataset_root=dataset_root,

            output_dir=output_dir,

            category_name=category_name,

            pixel_threshold=pixel_threshold,

            min_area=min_area

        )

        # =========================
        # 统一返回格式
        # =========================

        return {

            "algorithm":
                self.name,

            "defect_type":
                defect_type,

            "status":
                "success",

            "result":
                result

        }


if __name__ == "__main__":
    skill = PatchCoreSkill()

    result = skill.run(

        defect_type="surface_defect",

        params={

            "dataset_root":
                "data/stage4_anomaly/my_part_split",

            "output_dir":
                "outputs/stage4_anomaly"

        }

    )

    print(result)
