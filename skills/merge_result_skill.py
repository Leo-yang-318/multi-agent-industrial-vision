def normalize_task_type(application_goal: str) -> str:
    """
    将自然语言应用目标归一化成系统内部任务类型。
    """

    if not application_goal:
        return "unknown"

    if "表面" in application_goal and "缺陷" in application_goal:
        return "surface_defect_inspection"

    if "尺寸" in application_goal:
        return "dimension_inspection"

    if "分类" in application_goal:
        return "classification_inspection"

    if "装配" in application_goal:
        return "assembly_inspection"

    return "general_visual_inspection"


def merge_requirement_result(image_result: dict, text_result: dict) -> dict:
    """
    合并图像理解结果和文本解析结果，生成第一阶段需求结构化 JSON。
    """

    application_goal = text_result.get("application_goal", "未说明")

    requirement_json = {
        "stage": "stage1_requirement_structuring",

        "input": {
            "image_path": image_result.get("image_path"),
            "customer_text": text_result.get("raw_text")
        },

        "image_understanding": {
            "object_category": image_result.get("object_category", "未知"),
            "material": image_result.get("material", "未知"),
            "color": image_result.get("color", "未知"),
            "shape": image_result.get("shape", "未知"),
            "surface_features": image_result.get("surface_features", []),
            "possible_inspection_difficulty": image_result.get(
                "possible_inspection_difficulty", []
            ),
            "confidence": image_result.get("confidence", 0)
        },

        "customer_requirement": {
            "inspection_target": text_result.get("inspection_target", "未说明"),
            "defect_types": text_result.get("defect_types", ["未说明"]),
            "application_goal": application_goal,
            "output_requirement": text_result.get("output_requirement", {}),
            "constraints": text_result.get("constraints", {}),
            "missing_information": text_result.get("missing_information", [])
        },

        "structured_requirement": {
            "task_type": normalize_task_type(application_goal),

            "product_info": {
                "category": image_result.get("object_category", "未知"),
                "material": image_result.get("material", "未知"),
                "color": image_result.get("color", "未知"),
                "shape": image_result.get("shape", "未知"),
                "surface_features": image_result.get("surface_features", [])
            },

            "defect_info": {
                "inspection_area": text_result.get("inspection_target", "未说明"),
                "target_defects": text_result.get("defect_types", ["未说明"])
            },

            "output_info": text_result.get("output_requirement", {}),

            "project_constraints": text_result.get("constraints", {}),

            "missing_information": text_result.get("missing_information", []),

            "capture_condition": {
                "current_image_type": "natural_light_image",
                "need_stage2_capture_suggestion": True,
                "reason": "第一阶段只完成需求结构化，第二阶段需要根据工件外观、材质、缺陷类型和现场约束生成重拍或采集建议"
            }
        },

        "next_stage": "stage2_capture_suggestion"
    }

    return requirement_json