import re
def format_reward_think_answer(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?</think>\s*<tool>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</tool>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def format_reward_think_filtered_information(completions, completions_first, **kwargs):

    # 定义两个正则模式
    tool_pattern = r'.*?<tool>(.*?:.*?)</tool>'          # 原completion的检查格式
    step_answer_pattern = r'<think>(.*?)</think>.*?<answer>(.*?:.*?)</answer>'  # completion_first的检查格式

    # 处理completions_first的内容
    first_contents = [c[0]["content"] for c in completions_first]
    first_matches = [re.fullmatch(tool_pattern, content, re.DOTALL) for content in first_contents]

    # 处理原completions的内容
    original_contents = [c[0]["content"] for c in completions]
    original_matches = [re.fullmatch(step_answer_pattern, content, re.DOTALL) for content in original_contents]

    # 计算加权奖励
    return [
        0.3 * (1.0 if first else 0.0) + 0.7 * (1.0 if orig else 0.0)
        for first, orig in zip(first_matches, original_matches)
    ]
