from openai_api import generate_code
from engine import run_one_example

# 定义问题描述和测试用例
problems = [
    {
        "description": "Implement a function that returns the sum of two numbers.",
        "test_cases": [
            {"input": (1, 2), "output": 3},
            {"input": (10, 5), "output": 15},
            {"input": (-1, 1), "output": 0}
        ]
    },
    # 可以添加更多问题
]
success, failed = 0, 0
# 评估所有问题
for problem in problems:
    try:
        run_one_example(generate_code, problem)
        success += 1
    except:
        failed += 1

print("All tests have completed: %d cases passed, %d cases failed" %(success, failed))