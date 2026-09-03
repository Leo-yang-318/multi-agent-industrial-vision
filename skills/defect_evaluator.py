import json
from pathlib import Path
import pandas as pd


def load_metric_rule(
        rule_path="configs/defect_metric_rule.json"
):
    with open(
            rule_path,
            "r",
            encoding="utf-8"
    ) as f:
        rules = json.load(f)

    return rules


def load_results(
        csv_path
):
    df = pd.read_csv(csv_path)

    return df


def evaluate_by_type(
        defect_type,
        result,
        rules
):
    rule = rules.get(defect_type)

    if rule is None:
        return None

    output = {

        "defect_type": defect_type,

        "main_metrics":
            rule["main_metrics"],

        "reason":
            rule["reason"]

    }

    for metric in rule["main_metrics"]:

        if metric == "Recall":

            output["Recall"] = result["pixel_recall"]


        elif metric == "Score_Gap":

            output["Score_Gap"] = result["score_gap"]


        elif metric == "Defect_Mean_Score":

            output["Defect_Mean_Score"] = result["defect_mean_score"]


        elif metric == "IoU":

            output["IoU"] = result["iou"]


        elif metric == "Dice":

            output["Dice"] = result["dice"]

    return output


def evaluate_defect(result):
    """
    Agent统一评价入口

    输入:
        算法检测结果

    输出:
        指标结果
    """

    defect_type = result.get(
        "defect_type"
    )

    # 后续根据缺陷类型选择评价方法

    if defect_type == "scratch":

        metrics = {

            "Recall":
                result.get(
                    "Recall",
                    0
                )

        }


    else:

        metrics = {}

    return metrics


if __name__ == "__main__":

    rules = load_metric_rule()

    df = load_results(
        "outputs/stage4_anomaly/evaluation_results.csv"
    )

    for index, row in df.iterrows():
        result = evaluate_by_type(
            row["defect_type"],
            row,
            rules
        )

        print("================")

        print(result)
