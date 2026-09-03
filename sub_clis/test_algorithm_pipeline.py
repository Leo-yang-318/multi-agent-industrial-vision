from sub_clis.algorithm_agent import AlgorithmAgent
from utils.skill_executor import SkillExecutor


def main():
    defect_type = "scratch"
    image_path = r"D:\project\data\input_images\Image_20260320173604289.bmp"

    agent = AlgorithmAgent()
    executor = SkillExecutor()

    print("========== 1. 第一次算法选择 ==========")
    decision = agent.select_algorithm(defect_type)
    print("算法选择结果:", decision)

    print("========== 2. 第一次执行算法 ==========")
    first_result = executor.execute(
        skill_name=decision["skill"],
        image_path=image_path,
        defect_type=defect_type,
        params={}
    )

    print("第一次结果:")
    print(first_result)

    print("========== 3. 判断下一步动作 ==========")
    action = agent.decide_next_action(
        first_result,
        decision["backup_skill"]
    )

    print("下一步动作:")
    print(action)

    if action["action"] == "switch":
        print("========== 4. 开始执行备用算法 ==========")
        second_result = executor.execute(
            skill_name=action["next_skill"],
            image_path=image_path,
            defect_type=defect_type,
            params={}
        )

        print("第二次结果:")
        print(second_result)


if __name__ == "__main__":
    main()
