# 项目说明

本项目是“基于多智能体协同的工业视觉售前方案自动化评估系统”。

当前已完成第一阶段：需求结构化。

第一阶段输入：
- data/input_images/test.jpg：自然光工件图片
- data/input_text/customer_text.txt：客户需求文本

第一阶段输出：
- data/requirement_result.json
- state/state.json

当前 state.json 中的 current_stage 用于表示下一步应该执行的阶段。

## 当前代码结构

- main_cli.py：主 CLI，负责阶段调度和维护 state.json
- sub_cli/requirement_structuring_cli.py：第一阶段 Sub-CLI
- skills/image_understanding_skill.py：调用 Qwen-VL 识别图片
- skills/text_extract_skill.py：调用 Qwen 文本模型解析客户需求
- skills/merge_result_skill.py：合并图像结果和文本结果
- utils/json_utils.py：JSON 读写
- utils/file_utils.py：文件检查
- utils/state_manager.py：状态管理

## 运行命令

激活环境：

```powershell
cd /d D:\project
.\.venv\Scripts\Activate.ps1