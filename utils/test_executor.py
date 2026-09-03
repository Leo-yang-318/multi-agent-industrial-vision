import json

from utils.skill_executor import SkillExecutor

executor = SkillExecutor()

result = executor.execute(

    skill_name="traditional_defect_skill",

    image_path=
    r"D:\project\data\input_images\Image_20260320173604289.bmp",

    defect_type="scratch",

    params={}

)

print(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    )
)
