from utils.json_utils import read_json, write_json


class StateManager:
    """
    全局状态管理器。

    注意：
    1. state.json 由主 CLI 维护。
    2. Sub-CLI 只负责执行第一阶段任务并输出结果。
    3. 主 CLI 读取 Sub-CLI 输出后，再写入 state.json。
    """

    def __init__(self, state_path: str):
        self.state_path = state_path
        self.state = read_json(state_path)

    def init_stage1(self, image_path: str, text_path: str) -> None:
        self.state = {
            "current_stage": "stage1_requirement_structuring",
            "stage1": {
                "name": "需求结构化",
                "status": "running",
                "input": {
                    "image_path": image_path,
                    "customer_text_path": text_path
                },
                "output": {}
            }
        }
        self.save()

    def finish_stage1(self, requirement_json_path: str, requirement_result: dict) -> None:
        structured_requirement = requirement_result.get("structured_requirement", {})
        customer_requirement = requirement_result.get("customer_requirement", {})
        output_info = structured_requirement.get("output_info", {})
        capture_condition = structured_requirement.get("capture_condition", {})

        self.state["stage1"]["status"] = "finished"
        self.state["stage1"]["output"] = {
            "requirement_json_path": requirement_json_path,
            "summary": {
                "task_type": structured_requirement.get("task_type", "unknown"),
                "defect_types": customer_requirement.get("defect_types", []),
                "need_defect_location": output_info.get("need_defect_location", False),
                "need_stage2_capture_suggestion": capture_condition.get(
                    "need_stage2_capture_suggestion", False
                )
            }
        }
        self.state["current_stage"] = "stage2_capture_suggestion"
        self.state["next_stage"] = "stage2_capture_suggestion"
        self.state["stage1"]["error"] = None
        self.save()

    def fail_stage1(self, error_message: str) -> None:
        if "stage1" not in self.state:
            self.state["stage1"] = {
                "name": "需求结构化",
                "status": "failed",
                "input": {},
                "output": {}
            }

        self.state["stage1"]["status"] = "failed"
        self.state["stage1"]["error"] = error_message
        self.save()

    def save(self) -> None:
        write_json(self.state, self.state_path)
