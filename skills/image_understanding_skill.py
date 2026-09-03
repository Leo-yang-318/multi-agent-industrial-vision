import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
)


def encode_image_to_data_url(image_path: str) -> str:
    """
    将本地图片转成 base64 data URL，方便传给 OpenAI API。
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    elif suffix == ".webp":
        mime_type = "image/webp"
    else:
        raise ValueError(f"暂不支持该图片格式: {suffix}，请使用 jpg、jpeg、png 或 webp")

    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def safe_json_loads(text: str) -> dict:
    """
    尽量把模型返回内容解析为 JSON。
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return json.loads(text)


def run_image_understanding(image_path: str) -> dict:
    """
    图像理解 Skill：
    调用 ChatGPT 视觉能力识别工件的类别、材质、颜色、形状和表面特征。
    """

    image_data_url = encode_image_to_data_url(image_path)
    model = os.getenv("QWEN_VISION_MODEL", "qwen3-vl-plus")

    prompt = f"""
你是工业视觉售前评估系统中的“图像理解 Skill”。

现在输入的是一张自然光拍摄的工件图片。
请你只根据图片内容，识别工件的基础视觉信息。

请严格输出 JSON，不要输出解释文字，不要使用 Markdown 代码块。

JSON 格式必须如下：
{{
  "image_path": "{image_path}",
  "object_category": "工件类别，例如：金属零件、塑料件、电子元件、瓶盖、螺丝、外壳等；无法确定则写未知",
  "material": "材质，例如：金属、塑料、橡胶、玻璃、陶瓷、纸张、未知",
  "color": "主要颜色",
  "shape": "主要形状，例如：圆柱形、长方体、片状、环形、不规则形状等",
  "surface_features": ["表面特征1", "表面特征2"],
  "possible_inspection_difficulty": ["可能影响视觉检测的因素"],
  "confidence": 0.0
}}

要求：
1. 不要判断是否存在真实缺陷，只描述图片中可见外观。
2. 不要编造看不清的信息，看不清就写“未知”。
3. confidence 使用 0 到 1 的小数。
4. 只输出 JSON。
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
    )

    output_text = response.choices[0].message.content
    result = safe_json_loads(output_text)
    result["image_path"] = image_path

    return result