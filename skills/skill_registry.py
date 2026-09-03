"""
Skill 注册中心

作用：
1. 管理所有工业视觉检测算法
2. 根据缺陷类型找到对应Skill
3. 给Agent提供算法调用入口
"""

# ==========================
# 1. Skill注册表
# ==========================


SKILL_REGISTRY = {

    # ---------------------
    # 划痕检测
    # ---------------------

    "scratch":
        {

            "skill_name":
                "traditional_defect_skill",

            "algorithm_type":
                "traditional_cv",

            "description":
                "传统视觉划痕检测，适合线状缺陷"

        },

    # ---------------------
    # 漏孔检测
    # ---------------------

    "missing_hole":
        {

            "skill_name":
                "traditional_defect_skill",

            "algorithm_type":
                "traditional_cv",

            "description":
                "孔洞数量检测，适合结构型缺陷"

        },

    # ---------------------
    # 边缘缺料
    # ---------------------

    "edge_missing":
        {

            "skill_name":
                "traditional_defect_skill",

            "algorithm_type":
                "traditional_cv",

            "description":
                "边缘轮廓检测"

        },

    # ---------------------
    # 多料
    # ---------------------

    "extra_material":
        {

            "skill_name":
                "patchcore_skill",

            "algorithm_type":
                "anomaly_detection",

            "description":
                "区域异常检测"

        },

    # ---------------------
    # 切割错误
    # ---------------------

    "cutting_error":
        {

            "skill_name":
                "patchcore_skill",

            "algorithm_type":
                "anomaly_detection",

            "description":
                "结构异常检测"

        }

}


# ==========================
# 2. 根据缺陷类型获取Skill
# ==========================


def get_skill_by_defect(defect_type):
    """
    输入:
        scratch

    返回:
        对应检测Skill信息
    """

    if defect_type in SKILL_REGISTRY:

        return SKILL_REGISTRY[defect_type]


    else:

        return {

            "skill_name":
                "patchcore_skill",

            "algorithm_type":
                "anomaly_detection",

            "description":
                "未知缺陷，使用通用异常检测"

        }


# ==========================
# 3. 查看所有Skill
# ==========================


def list_all_skills():
    return SKILL_REGISTRY


# ==========================
# 测试
# ==========================


if __name__ == "__main__":
    result = get_skill_by_defect(
        "scratch"
    )

    print(result)
