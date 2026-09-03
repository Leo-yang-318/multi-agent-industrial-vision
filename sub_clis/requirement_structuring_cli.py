import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import traceback

from skills.image_understanding_skill import run_image_understanding
from skills.text_extract_skill import run_text_extract
from skills.merge_result_skill import merge_requirement_result
from utils.file_utils import ensure_file_exists
from utils.json_utils import write_json


def main():
    parser = argparse.ArgumentParser(description="第一阶段：需求结构化 Sub-CLI")

    parser.add_argument(
        "--image",
        required=True,
        help="自然光工件图片路径"
    )

    parser.add_argument(
        "--text",
        required=True,
        help="客户需求文本路径"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="需求结构化 JSON 输出路径"
    )

    args = parser.parse_args()

    try:
        ensure_file_exists(args.image)
        ensure_file_exists(args.text)

        print("正在调用图像理解 Skill...")
        image_result = run_image_understanding(args.image)

        print("正在调用文本需求解析 Skill...")
        text_result = run_text_extract(args.text)

        print("正在合并图像结果和文本结果...")
        requirement_result = merge_requirement_result(
            image_result=image_result,
            text_result=text_result
        )

        write_json(requirement_result, args.output)

        print("需求结构化 Sub-CLI 执行完成")
        print(f"输出文件: {args.output}")

    except Exception as e:
        print("需求结构化 Sub-CLI 执行失败")
        print(str(e))
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()