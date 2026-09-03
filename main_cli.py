import argparse
import subprocess
import sys

from utils.file_utils import ensure_file_exists
from utils.json_utils import read_json
from utils.state_manager import StateManager
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="工业视觉售前自动化评估系统主 CLI")

    parser.add_argument(
        "--image",
        default="data/input_images/test.jpg",
        help="自然光工件图片路径"
    )

    parser.add_argument(
        "--text",
        default="data/input_text/customer_text.txt",
        help="客户需求文本路径"
    )

    parser.add_argument(
        "--state",
        default="state/state.json",
        help="全局状态文件路径"
    )

    parser.add_argument(
        "--stage1-output",
        default="data/requirement_result.json",
        help="第一阶段需求结构化结果输出路径"
    )

    args = parser.parse_args()

    ensure_file_exists(args.image)
    ensure_file_exists(args.text)

    state_manager = StateManager(args.state)

    state_manager.init_stage1(
        image_path=args.image,
        text_path=args.text
    )

    project_root = Path(__file__).resolve().parent

    command = [
        sys.executable,
        str(project_root / "sub_clis" / "requirement_structuring_cli.py"),
        "--image",
        args.image,
        "--text",
        args.text,
        "--output",
        args.stage1_output
    ]

    subprocess.run(command, check=True, cwd=project_root)

    try:
        print("主 CLI：正在调用需求结构化 Sub-CLI...")
        subprocess.run(command, check=True)

        requirement_result = read_json(args.stage1_output)

        state_manager.finish_stage1(
            requirement_json_path=args.stage1_output,
            requirement_result=requirement_result
        )

        print("第一阶段完成：需求结构化结果已写入 state/state.json")
        print("下一阶段：stage2_capture_suggestion")

    except Exception as e:
        state_manager.fail_stage1(str(e))
        print("第一阶段执行失败，错误已写入 state/state.json")
        raise


if __name__ == "__main__":
    main()