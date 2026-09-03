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


def run_text_extract(text_path: str) -> dict:
    """
    文本需求解析 Skill：
    调用 ChatGPT 从客户需求文本中提取结构化信息。
    """

    path = Path(text_path)

    if not path.exists():
        raise FileNotFoundError(f"客户需求文本不存在: {text_path}")

    customer_text = path.read_text(encoding="utf-8").strip()

    if not customer_text:
        raise ValueError("客户需求文本为空，请先填写 data/input_text/customer_text.txt")

    model = os.getenv("QWEN_TEXT_MODEL", "qwen-plus")

    prompt = f"""
你是工业视觉售前评估系统中的“客户需求结构化 Skill”。

请从下面客户需求文本中提取结构化信息。

客户需求文本：
{customer_text}

请严格输出 JSON，不要输出解释文字，不要使用 Markdown 代码块。

JSON 格式必须如下：
{{
  "raw_text": "原始客户需求文本",
  "inspection_target": "检测对象，例如：零件表面、瓶盖外观、焊点区域等",
  "defect_types": ["缺陷类型，例如：划痕、污渍、缺口、裂纹、变形、毛刺、色差等"],
  "application_goal": "应用目标，例如：表面缺陷检测、尺寸检测、分类检测、装配完整性检测等",
  "output_requirement": {{
    "need_defect_judgement": true,
    "need_defect_location": true,
    "need_defect_classification": true,
    "need_visual_result": true
  }},
  "constraints": {{
    "shooting_condition": "拍摄条件，例如：自然光样品图、现场相机图、客户未说明等",
    "speed_requirement": "节拍或速度要求，未说明则写未说明",
    "accuracy_requirement": "精度、误检率、漏检率要求，未说明则写未说明",
    "environment_requirement": "现场环境要求，未说明则写未说明",
    "customer_text_complete": true
  }},
  "missing_information": ["客户当前未提供但后续需要补充的信息"]
}}

要求：
1. 只根据客户文本提取，不要编造。
2. 没有提到的内容写“未说明”。
3. defect_types 如果未明确，写 ["未说明"]。
4. missing_information 要指出后续售前评估需要补充什么。
5. 只输出 JSON。
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    output_text = response.choices[0].message.content
    result = safe_json_loads(output_text)

    return result