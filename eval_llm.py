import time
import json
from collections import defaultdict
import re
import numpy as np
import copy

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution

def compute_accuracy(gt, pred_solution):
    answers = solve_math_problems(gt)

    if answers is None:
        return None, None

    if type(pred_solution) == dict:
        pred_answers = {}
        for k, v in pred_solution.items():
            pred_answer = parse_answer(v)
            if pred_answer is None:
                pred_answer = solve_math_problems(v)
            pred_answer = '0' if pred_answer == '.' else pred_answer
            pred_answers[k] = 1 if pred_answer is not None and float(answers) == float(pred_answer) else 0

        # print("pred_answers: ", pred_answers)
        pred_answer = most_frequent(list(pred_answers.values()))

        return pred_answer, pred_answers
        # print("pred answer: ", pred_answer)
        # pred_answer = pred_answers[0]
    elif type(pred_solution) == str:
        pred_answer = parse_answer(pred_solution)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

        if pred_answer is None:
            print('Unrecognized answer')
            return 0, None

        # try:
        pred_answer = '0' if pred_answer == '.' or pred_answer == '' else pred_answer
        return float(answers) == float(pred_answer), None
    # except:
    #     import pdb
    #     pdb.set_trace()
    #     print(pred_solution)

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

def eval_llm(result_dirs=['/content/drive/MyDrive/results/gsm_Llama-2-7b_test_100.json'], num_questions=None):
    accuracies = []
    llm_accuracies = defaultdict(list)
    idx = 0
    for result_dir in result_dirs:
        print(f'evaluating {result_dir} ...')
        response_dict = json.load(open(result_dir, "r"))

        for i, (k, v) in enumerate(response_dict.items()):
            if num_questions is not None:
                if idx < num_questions[0] or idx >= num_questions[1]: continue
            if type(v) is list:
                response, gt = v
            else:
                response = copy.deepcopy(v)
                gt = v['answer']
                del response['answer']

            accurate, answers = compute_accuracy(gt, response)
            if answers is not None:
                for k, v in answers.items():
                    llm_accuracies[k].append(v if v is not None else 0)
            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                accuracies.append(0)
                print('Accurate is None!')

            idx += 1

    print("accuracies: {:.4f} +/- {:.4f}".format(np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5)))
    if len(llm_accuracies) > 0:
        for k, v in llm_accuracies.items():
            print("{} accuracies: {:.4f} +/- {:.4f}".format(k, np.mean(v), np.std(v) / (len(v) ** 0.5)))