import json
import glob
import tqdm
import torch
import json
from tqdm import tqdm
import random
from eval.answer_reward_utils import evaluate_example



def eval(input_path, OUTPUT_PATH, TEST_DATASETS):
    file_list = sorted(glob.glob(f"{input_path}*.json"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    question_list = []
    result_list = []
    solution_list = []
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            question_list.extend(data['question'])
            result_list.extend(data['results'])
            solution_list.extend(data['solution'])
    final_output = []
    correct_number = 0
    pbar = tqdm(total=len(result_list), desc="Evaluating results")
    for model_output, solution, question in  zip(result_list, solution_list, question_list):
        correct = evaluate_example(model_output, solution, question)
        correct_number += correct
        final_output.append({
            'question': question,
            'ground_truth': solution,
            'model_output': model_output,
            'correct': correct
        })
        pbar.update(1)
    pbar.close()
    accuracy = correct_number / len(result_list) * 100
    output_path = OUTPUT_PATH.format(DATASET=TEST_DATASETS[0])
    with open(output_path, "w") as f:
        json.dump({'accuracy': accuracy, 'results': final_output}, f, indent=2)

def main():
    TEST_DATASETS = ['infoseek_test']
    input_path = "results_"
    random.seed(42)
    
    
    OUTPUT_PATH = "result.json"
    eval(input_path, OUTPUT_PATH, TEST_DATASETS)

if __name__ == "__main__":
    main()
