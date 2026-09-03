import pandas as pd


def analyze_defect_type(
        csv_path="outputs/stage4_anomaly/evaluation_results.csv"
):
    df = pd.read_csv(csv_path)

    print("\n====== 缺陷分类统计 ======")

    result = df.groupby(
        "defect_type"
    ).agg(
        {
            "score_gap": "mean",
            "defect_mean_score": "mean",
            "pixel_recall": "mean",
            "iou": "mean",
            "dice": "mean",
        }
    )

    print(result)


if __name__ == "__main__":
    analyze_defect_type()
