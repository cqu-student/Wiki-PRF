import os, sys
sys.path.append(os.getcwd())
from datetime import datetime
import re
from .reward.answer_reward_utils import evaluate_example
from .reward.answer_reward_utils_evqa import evaluate_example_evqa


def answer_reward_rag(completions, solution, problem, **kwargs):
    candidate_list = [completion[0]["content"] for completion in completions]
    reference_list = solution
    question_list = problem
    rewards = []

    
    for candidate, reference, question in zip(candidate_list, reference_list, question_list):
        reward = 0.0
        reward = evaluate_example(candidate, reference, question)
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"candidate: {candidate}\n")
                f.write(f"reference: {reference}\n")
    return rewards


def answer_reward_rag(completions, solution, problem, **kwargs):
    candidate_list = [completion[0]["content"] for completion in completions]
    reference_list = solution
    question_list = problem
    rewards = []

    
    for candidate, reference, question in zip(candidate_list, reference_list, question_list):
        reward = 0.0
        reward = evaluate_example(candidate, reference, question)
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"candidate: {candidate}\n")
                f.write(f"reference: {reference}\n")
    return rewards

def answer_reward_rag_evqa(completions, solution, problem, problem_type, **kwargs):
    candidate_list = [completion[0]["content"] for completion in completions]
    reference_list = solution
    question_list = problem
    question_type_list = problem_type
    rewards = []

    
    for candidate, reference, question, question_type in zip(candidate_list, reference_list, question_list, question_type_list):
        reward = 0.0
        reward = evaluate_example_evqa(candidate, reference, question, question_type)
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"candidate: {candidate}\n")
                f.write(f"reference: {reference}\n")
    return rewards
    


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    completions = [[{'role': 'assistant', 'content': 'This lake belongs to both North Macedonia and Albania.'}], [{'role': 'assistant', 'content': 'North Macedonia'}], [{'role': 'assistant', 'content': 'The lake in the picture is Lake Ohrid, which belongs to North Macedonia.'}], [{'role': 'assistant', 'content': 'Macedonia'}]]
    problem = ['What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?', 'What country does this lake belong to?']
    solution = [['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania'], ['North Macedonia', 'Albania']]
    print(answer_reward_rag(completions, solution, problem=problem))


