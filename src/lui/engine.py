import re


def extract_code(message):
    match = re.search("(def .+?)```", message, flags=re.DOTALL)
    if match:
        code = match.group(1)
        return code
    return message


# 测试生成代码
def run_one_example(fun, problem):
    description = problem["description"]
    test_cases = problem["test_cases"]
    response = fun(description)
    pure_code = extract_code(response)
    # 将生成的代码作为函数进行测试
    exec(pure_code)
    func = locals()["solution_function"]  # 假设生成的代码定义了一个名为solution_function的函数
    for test in test_cases:
        input_data = test["input"]
        expected_output = test["output"]
        assert func(*input_data) == expected_output, f"Failed test: {input_data} != {expected_output}"