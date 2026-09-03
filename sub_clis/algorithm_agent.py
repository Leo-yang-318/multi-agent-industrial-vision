import json
from pathlib import Path


class AlgorithmAgent:

    def __init__(self):
        config_path = Path(
            "configs/algorithm_library.json"
        )

        with open(
                config_path,
                "r",
                encoding="utf-8"
        ) as f:
            self.algorithm_library = json.load(f)

    def evaluate_algorithm(
            self,
            result
    ):

        recall = result["metrics"].get(
            "Recall",
            0
        )

        if recall < 0.7:

            return {

                "action":
                    "change_algorithm",

                "next_skill":
                    "patchcore_skill",

                "reason":
                    "当前算法召回率不足"

            }


        else:

            return {

                "action":
                    "keep",

                "reason":
                    "当前算法满足要求"

            }

    def select_algorithm(
            self,
            defect_type
    ):
        """
        根据缺陷类型选择算法
        """

        if defect_type not in self.algorithm_library:
            return {

                "skill":
                    "patchcore_skill",

                "reason":
                    "未知缺陷，使用通用异常检测"

            }

        info = self.algorithm_library[defect_type]

        return {

            "skill":
                info["preferred_skill"],

            "backup_skill":
                info["backup_skill"],

            "reason":
                info["reason"]

        }

    def decide_next_action(
            self,
            result,
            backup_skill
    ):

        """
        根据检测结果决定是否切换算法
        """

        metrics = result.get(
            "metrics",
            {}
        )

        recall = metrics.get(
            "Recall",
            0
        )

        # 判断标准
        if recall < 0.7:

            return {

                "action":
                    "switch",

                "next_skill":
                    backup_skill,

                "reason":
                    "当前算法召回率不足"

            }


        else:

            return {

                "action":
                    "keep",

                "next_skill":
                    None,

                "reason":
                    "当前算法满足要求"

            }


if __name__ == "__main__":
    agent = AlgorithmAgent()

    result = agent.select_algorithm(
        "scratch"
    )

    print(result)
