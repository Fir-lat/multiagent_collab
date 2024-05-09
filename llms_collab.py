import asyncio
import fastapi_poe as fp
from tqdm import tqdm
from datasets import load_dataset
import argparse
import time
import json
import numpy as np
from collections import defaultdict
import re
import copy

class Config():
    def __init__(
            self,
            data_split='test',
            num_questions=(0, 100),
            batch_size=1,
            llm=['GPT-3.5-Turbo', 'GPT-3.5-Turbo', 'GPT-3.5-Turbo'],
            dataset_path='datasets/gsm8k/main',
            num_agents=3,
            rounds=2,
            api_key='ab9MG8v8wwwilE1DBM4EIHA1pffo-N9Bru0LCJ7B_zo',
            check_freq=10,
            time_delay=5
    ):
        self.data_split = data_split # choices=['test', 'train']
        self.num_questions = num_questions
        self.batch_size = batch_size
        self.llm = llm
        self.dataset_path = dataset_path
        self.num_agents = num_agents
        self.rounds = rounds
        self.api_key = api_key
        self.check_freq = check_freq
        self.time_delay = time_delay

contexts_prefix = 'Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.'
contexts_suffix = '{} Please use solutions from other agents to check your own solution, and provide the final answer in a single numerical number, in the form\\boxed{{answer}}'

teacher_contexts_suffix = 'Please modify your solution according to these suggestions: "{}" Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.'

student_contexts_init = 'Here is a math problem: {} This is an agent\' s solution: "{}" Please check the solution. If there is anything wrong, list it.'
student_contexts_suffix = 'The agent\'s new solution: "{}" Please check the solution. If there is anything wrong, list it.'
teacher_student_contexts_suffix = 'Now please give me your own answer to the problem according to previous discussion. Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.'

def prepare_prompt(messages):
    prompt = fp.QueryRequest(
            query=messages,
            user_id="",
            conversation_id="",
            message_id="",
            version='1.0',
            type="query",
        )
    return prompt

# Create an asynchronous function to encapsulate the async for loop
def get_responses(api_key, messages, bot_name, tries=10):
    cnt = 0
    result = None
    while cnt < tries:
        try:
            result = fp.get_final_response(request=messages, bot_name=bot_name, api_key=api_key, retry_sleep_time=60)
            break
        except fp.BotError as e:
            print("Caught BotError exception:", e)
            time.sleep(60)
            # Handle the exception here
        cnt += 1
    return result


def construct_debate(contexts, agent_name, round):
    other_agent_solution = []
    for k, v in contexts.items():
        if k == agent_name:
            continue
        other_agent_solution.append(f'''One other agent's solution: \"{v[2 * round - 1]}\"\n''' )
    return contexts_suffix.format(''.join(other_agent_solution))

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

async def simple_test(config):
    dataset = load_dataset(config.dataset_path, split=config.data_split)
    dataset = dataset.select(range(config.num_questions[0], config.num_questions[1]))
    diff_agents = set(config.llm)
    for agent in diff_agents:
        generated_contexts = defaultdict(list)
        generated_description = {}

        for i, data in enumerate(tqdm(dataset)):
            question = data['question']
            answer = data['answer']
            generated_contexts[question].append(fp.ProtocolMessage(role="user", content=contexts_prefix.format(question)))
            prompt = prepare_prompt(generated_contexts[question])
            response = get_responses(config.api_key, prompt, agent)
            time.sleep(config.time_delay)
            result = await response
            generated_description[question] = (result, answer)
            if i > 0 and (i + 1) % config.check_freq == 0:
                json.dump(generated_description, open("results/answers/gsm_{}_{}_{}.json".format(agent, config.data_split, (config.num_questions[0], i + 1)), "w"))
        if config.num_questions[1] - config.num_questions[0] % config.check_freq != 0:
            json.dump(generated_description, open("results/gsm_{}_{}_{}.json".format(agent, config.data_split, config.num_questions), "w"))
        contexts = {}
        for k, v in generated_contexts.items():
            context = []
            for conversation in v:
                context.append(conversation.role + ": " + conversation.content)
            contexts[k] = '\n'.join(context)
        json.dump(contexts, open("results/contexts/gsm_{}_{}_{}.json".format(agent, config.data_split, config.num_questions), "w"))
    return generated_contexts, generated_description

async def debate_test(config):
    dataset = load_dataset(config.dataset_path, split=config.data_split)
    dataset = dataset.select(range(config.num_questions[0], config.num_questions[1]))
    generated_contexts = {}
    generated_description = {}

    for i, data in enumerate(tqdm(dataset)):
        question = data['question']
        answer = data['answer']
        generated_contexts[question] = defaultdict(list)
        generated_description[question] = {}
        for round in range(config.rounds):
            for k, agent in enumerate(config.llm):
                agent_name = f'{agent}_{k}'
                if round == 0:
                    round_prompt = fp.ProtocolMessage(role="user", content=contexts_prefix.format(question))
                else:
                    prompt = construct_debate(generated_contexts[question], agent_name, round)
                    round_prompt = fp.ProtocolMessage(role="user", content=prompt)
                generated_contexts[question][agent_name].append(round_prompt)
                prompt = prepare_prompt(generated_contexts[question][agent_name])
                response = get_responses(config.api_key, prompt, agent)
                time.sleep(config.time_delay)
                result = await response
                generated_contexts[question][agent_name].append(fp.ProtocolMessage(role="bot", content=result))
        for j, agent in enumerate(config.llm):
            agent_name = f'{agent}_{j}'
            generated_description[question][agent_name] = generated_contexts[question][agent_name][-1].content
        generated_description[question]['answer'] = answer
        if i > 0 and (i + 1) % config.check_freq == 0:
            json.dump(generated_description, open("results/answers/gsm_{}*{}_{}_{}_{}.json".format(config.llm, config.num_agents, config.data_split, (config.num_questions[0], i + 1), config.rounds), "w"))
    if config.num_questions[1] - config.num_questions[0] % config.check_freq != 0:
        json.dump(generated_description, open("results/gsm_{}*{}_{}_{}_{}.json".format(config.llm, config.num_agents, config.data_split, config.num_questions, config.rounds), "w"))
    contexts = {}
    for k, v in generated_contexts.items():
        contexts[k] = {}
        for agent, conversations in v.items():
            context = []
            for conversation in conversations:
                context.append(conversation.role + ": " + conversation.content)
            contexts[k][agent] = '\n'.join(context)
    json.dump(contexts, open("results/contexts/gsm_{}*{}_{}_{}_{}.json".format(config.llm, config.num_agents, config.data_split, config.num_questions, config.rounds), "w"))
    return generated_contexts, generated_description

async def chain_test(config):
    '''
        Implementation of interaction chain of LLM agents
    '''
    dataset = load_dataset(config.dataset_path, split=config.data_split)
    dataset = dataset.select(range(config.num_questions[0], config.num_questions[1]))
    generated_contexts = {}
    generated_description = {}

    for i, data in enumerate(tqdm(dataset)):
        question = data['question']
        answer = data['answer']
        generated_contexts[question] = defaultdict(list)
        generated_description[question] = {}
        for k, teacher in enumerate(config.llm):
            student = config.llm[(k + 1) % len(config.llm)]
            teacher_agent = f'{teacher}_{k}'
            student_agent = f'{student}_{k + 1}'
            for round in range(config.rounds - 1):
                if round == 0:
                    if teacher == 0:
                        teacher_prompt = fp.ProtocolMessage(role="user", content=contexts_prefix.format(question))
                    else:
                        teacher_prompt = fp.ProtocolMessage(role="user", content=teacher_student_contexts_suffix)
                else:
                    teacher_prompt = fp.ProtocolMessage(role="user", content=teacher_contexts_suffix.format(generated_contexts[question][student_agent][-1].content))
                generated_contexts[question][teacher_agent].append(teacher_prompt)
                prompt = prepare_prompt(generated_contexts[question][teacher_agent])
                response = get_responses(config.api_key, prompt, teacher)
                time.sleep(config.time_delay)
                result = await response
                generated_contexts[question][teacher_agent].append(fp.ProtocolMessage(role="bot", content=result))

                if round == 0:
                    student_prompt = fp.ProtocolMessage(role="user", content=student_contexts_init.format(question, result))
                else:
                    student_prompt = fp.ProtocolMessage(role="user", content=student_contexts_suffix.format(generated_contexts[question][teacher_agent][-1].content))
                generated_contexts[question][student_agent].append(student_prompt)
                prompt = prepare_prompt(generated_contexts[question][student_agent])
                response = get_responses(config.api_key, prompt, student)
                time.sleep(config.time_delay)
                result = await response
                generated_contexts[question][student_agent].append(fp.ProtocolMessage(role="bot", content=result))

            teacher_prompt = fp.ProtocolMessage(role="user", content=teacher_contexts_suffix.format(generated_contexts[question][student_agent][-1].content))
            generated_contexts[question][teacher_agent].append(teacher_prompt)
            prompt = prepare_prompt(generated_contexts[question][teacher_agent])
            response = get_responses(config.api_key, prompt, teacher)
            time.sleep(config.time_delay)
            result = await response
            generated_contexts[question][teacher_agent].append(fp.ProtocolMessage(role="bot", content=result))

        for j, agent in enumerate(config.llm):
            agent_name = f'{agent}_{j}'
            generated_description[question][agent_name] = generated_contexts[question][agent_name][-1].content
        generated_description[question]['answer'] = answer
        if i > 0 and (i + 1) % config.check_freq == 0:
            json.dump(generated_description, open("results/answers/gsm_{}>{}_{}_{}_{}.json".format(config.llm, config.num_agents, config.data_split, (config.num_questions[0], i + 1), config.rounds), "w"))
    if config.num_questions[1] - config.num_questions[0] % config.check_freq != 0:
        json.dump(generated_description, open("results/gsm_{}>{}_{}_{}_{}.json".format(config.llm, config.num_agents, config.data_split, config.num_questions, config.rounds), "w"))
    contexts = {}
    for k, v in generated_contexts.items():
        contexts[k] = {}
        for agent, conversations in v.items():
            context = []
            for conversation in conversations:
                context.append(conversation.role + ": " + conversation.content)
            contexts[k][agent] = '\n'.join(context)
    json.dump(contexts, open("results/contexts/gsm_{}>{}_{}_{}_{}.json".format(config.llm, config.num_agents, config.data_split, config.num_questions, config.rounds), "w"))
    return generated_contexts, generated_description


async def main():
    config1 = Config(
        data_split='test',
        num_questions=(0, 500),
        batch_size=1,
        llm=['Gemini-1.0-Pro', 'GPT-3.5-Turbo', 'Qwen-72b-Chat'], # Qwen-72b-Chat
        dataset_path='datasets/gsm8k/main',
        num_agents=3,
        rounds=2,
        api_key='ab9MG8v8wwwilE1DBM4EIHA1pffo-N9Bru0LCJ7B_zo',
        check_freq=10,
        time_delay=5
    )

    contexts, description = await simple_test(config1)

if __name__ == '__main__':
    main()