"""
Skill执行器

作用：
根据AlgorithmAgent返回的Skill名称
自动找到对应Skill并执行
"""
from skills.defect_evaluator import evaluate_defect
from skills.traditional_defect_skill import TraditionalDefectSkill
from skills.anomaly_patchcore_skill import PatchCoreSkill


class SkillExecutor:

    def __init__(self):
        # Skill注册表

        self.skills = {

            "traditional_defect_skill":
                TraditionalDefectSkill(),

            "patchcore_skill":
                PatchCoreSkill()

        }

    def execute(
            self,
            skill_name,
            image_path,
            defect_type,
            params=None
    ):
        """
        执行指定Skill

        """

        if skill_name not in self.skills:
            return {

                "status":
                    "failed",

                "message":
                    f"不存在Skill:{skill_name}"

            }

        skill = self.skills[skill_name]

        result = skill.run(

            image_path=image_path,

            defect_type=defect_type,

            params=params

        )
        evaluation = evaluate_defect(

            result

        )
        result["metrics"] = evaluation

        return result


if __name__ == "__main__":
    executor = SkillExecutor()

    result = executor.execute(

        skill_name=
        "traditional_defect_skill",

        image_path=
        "data/input_images/test.bmp",

        defect_type=
        "scratch",

        params={}

    )

    print(result)
