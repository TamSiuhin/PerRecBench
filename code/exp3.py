import json
from rank_bm25 import BM25Okapi
from openai import OpenAI
import os
from tqdm import tqdm
import random
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau

import sys

sys.path.append('../')

import call_api
from utils import extract_first_number

client = OpenAI()

def make_template(index):
    with open(f'templates_point/3dai/3dai{index}.txt', 'r') as f:
        template = f.read()
    return template

def encode_query(id, item_dict):
    item = item_dict[id]
    prompt = item['title']
    return prompt

def encode_history(behavior, item_dict):
    item = item_dict[behavior['parent_asin']]
    prompt = item['title']
    return prompt

def encode_example(template, items, Qs, A):
    if type(A) is str:
        template = template[:-3]
        histories = [encode_history(behavior, items) for behavior in Qs]
        histories = ', '.join(histories)
        query = A
        prompt = template.format(histories, query)
    else:
        histories = [encode_history(behavior, items) for behavior in Qs]
        histories = ', '.join(histories)
        query = encode_query(A['parent_asin'], items)
        label = A['rating']
        prompt = template.format(histories, query, label)
    return prompt

def retrieval(corpus, query, n):
    tokenized_corpus = [doc.split() for doc in corpus]
    tokenized_query = query.split(" ")
    bm25 = BM25Okapi(tokenized_corpus)
    profiles = bm25.get_top_n(tokenized_query, corpus, n)
    return profiles

# def GetAnswer(prompt):
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo-0125",
#         messages=[{"role": "system", "content": "You are a book recommender system now and designed to output just integer numbers."},
#                   {"role": "user", "content": prompt}])
#     out = response.choices[0].message.content
#     return out

def get_input(path, level):
    with open(f'../data/{path}/parentasin2item.json') as f:
        items = json.load(f)
    with open(f'../data/{path}/userid2history.json') as f:
        users = json.load(f)

    if 'processed' in path:
        with open(f'../data/{path}/item2group_avg_diff_sample-{level}.json') as f:
            groups = json.load(f)
    else:
        with open(f'../data/{path}/sample_behavior.json') as f:
            groups = json.load(f)
    return items, users, groups

def make_shot_point(index, behavior_all, history_idx, items, retrieval_num, shot):
    with open(f'templates_point/3dai/3dai{index}.txt', 'r') as f:
        template = f.read()
    
    corpus = []
    for behavior in behavior_all[:history_idx]:
        history = encode_history(behavior, items)
        corpus.append(history)

    shot_prompt = ''
    sampled_idx = random.sample(list(range(retrieval_num, len(corpus))), shot)

    for i in sampled_idx:
        
        sampled_behavior = behavior_all[i]

        retrieved_corpus = corpus[:i]
        retrieved_his = retrieval(retrieved_corpus, corpus[i], retrieval_num)

        # his_temp = "<|The Start of User History|>\n{}\n<|The End of User History|>\n\n" * len(retrieved_his)
        
        # his = his_temp.format(*retrieved_his)

        query = encode_query(sampled_behavior['parent_asin'], items)

        template_example = template.format(*retrieved_his, query)
        
        template_example += '{}\n\n'.format(sampled_behavior['rating'])

        shot_prompt += template_example

    return shot_prompt


def main3(index, rn, mode, count, model_id, cat, temp, shot=0, level='easy'):
    system_prompt ="You are a product recommender system now and designed to output just numbers."
    retrieval_num = rn
    items, users, groups = get_input(f"{cat}/{cat}-{mode.split('-')[0]}", level)
    template = make_template(index)

    if mode == 'processed-point':
        output = {}
        input = {}
        for key in tqdm(groups.keys()):
            output[key] = []
            input[key] = []
            q = encode_query(key, items)
            for group in groups[key]:
                input[key].append([])
                output[key].append([])
                for user in group:
                    behaviors = users[user['user_id']]
                    corpus = []
                    # for i in range(0, user['history_len']-retrieval_num, retrieval_num+1):
                    #     Qs = behaviors[i:i+retrieval_num]
                    #     A = behaviors[i+retrieval_num]
                    #     prompt = encode_example(template, items, Qs, A)
                    #     corpus.append(prompt)

                    for behavior in behaviors[:user['history_len']]:
                        history = encode_history(behavior, items)
                        corpus.append(history)

                    retr_his = retrieval(corpus, q, retrieval_num)
                    retr_his = ", ".join(retr_his)

                    prompt = template.format(retr_his, q)
                    
                    if shot != 0:
                        prompt_shot = make_shot_point(index, behaviors, user['history_len'], items, retrieval_num, shot)
                        prompt = prompt_shot + prompt

                    # print(prompt)
                    # print('-------------------------------')

                    # Qs = behaviors[-retrieval_num:]
                    # retr_his.append(encode_example(template, items, Qs, q))
                    # prompt = '\n\n'.join(retr_his)
                    # print(prompt)
                    # print('---------------------------------------------')
                    # while True:
                    #     print('--------------------------------')
                    #     print(prompt)
                    #     out = call_api.GetAnswer(prompt, model_id, system_prompt, gen_len=10)
                    #     out = extract_first_number(out)
                    #     print('--------------------------------')
                    #     print(out)
                    #     try:
                    #         if out[-1] == '.':
                    #             out = float(out[:-1])
                    #         else:
                    #             out = float(out)
                    #         break
                    #     except:
                    #         # print(out)
                            # ...


                    message = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    engine = 'openai'

                    while True:
                        out = call_api.GetAnswer(
                            model_id=model_id, 
                            engine=engine,
                            temp=temp, 
                            message=message,
                            sys_prompt=system_prompt,
                            n=1
                        )[0]
                        print(out)
                        try:
                            out = extract_first_number(out)
                            out = float(out)
                            break
                        except:
                            ...

                    output[key][-1].append(out)
                    input[key][-1].append(prompt)
    else:
        output = []
        input = []  
        for behavior in tqdm(groups):
            key = behavior["parent_asin"]
            q = encode_query(key, items)
            behaviors = users[behavior['user_id']]
            corpus = []
            # for i in range(0, behavior['history_len']-retrieval_num, retrieval_num+1):
            #     Qs = behaviors[i:i+retrieval_num]
            #     A = behaviors[i+retrieval_num]
            #     prompt = encode_example(template, items, Qs, A)
            #     corpus.append(prompt)

            for b in behaviors[:behavior['history_len']]:
                history = encode_history(b, items)
                corpus.append(history)


            retr_his = retrieval(corpus, q, retrieval_num)
            # Qs = behaviors[-retrieval_num:]
            # retr_his.append(encode_example(template, items, Qs, q))
            # prompt = '\n\n'.join(retr_his)
            
            retr_his = ", ".join(retr_his)

            prompt = template.format(retr_his, q)
            
            if shot != 0:
                prompt_shot = make_shot_point(index, behaviors, behavior['history_len'], items, retrieval_num, shot)
                prompt = prompt_shot + prompt
            
            # print(prompt)
            # print('-------------------------------')
            # while True:   
            #     out = call_api.GetAnswer(prompt, model_id, system_prompt, gen_len=10)
            #     out = extract_first_number(out)
            #     try:
            #         if out[-1] == '.':
            #             out = float(out[:-1])
            #         else:
            #             out = float(out)
            #         break
            #     except:
            #         ...

            message = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
            engine = 'openai'

            while True:
                out = call_api.GetAnswer(
                    model_id=model_id, 
                    engine=engine,
                    temp=temp, 
                    message=message,
                    sys_prompt=system_prompt,
                    n=1
                    )[0]
                try:
                    print(out)
                    out = extract_first_number(out)
                    out = float(out)
                    break
                except:
                    ...

            output.append(out)
            input.append(prompt)

    # with open(f"result/{retrieval_num}/{mode}/input/{count}.json", 'w') as f:
    #     json.dump(input, f)
    # with open(f"result/{retrieval_num}/{mode}/output/{count}.json", 'w') as f:
    #     json.dump(output, f)

    directory_in = f"result-correlation/{model_id}/{shot}shot/{cat}/exp3/k{retrieval_num}/{mode}/input/"
    if not os.path.exists(directory_in):
        # If it doesn't exist, create the directory
        os.makedirs(directory_in)

    directory_out = f"result-correlation/{model_id}/{shot}shot/{cat}/exp3/k{retrieval_num}/{mode}/output/"
    if not os.path.exists(directory_out):
        # If it doesn't exist, create the directory
        os.makedirs(directory_out)

    with open(os.path.join(directory_in, f"{index}.json"), 'w') as f:
        json.dump(input, f)

    with open(os.path.join(directory_out, f"{index}.json"), 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Parser for LoRA")
    parser.add_argument('--model_id', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--kind', type=int, default=0)
    parser.add_argument('--cat', type=str, default='all')
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--shot', type=int, default=0)
    parser.add_argument('--mode', type=str, default='traditional')
    parser.add_argument('--retrieval_k', type=int, default=4)
    parser.add_argument('--level', type=str, default='easy')

    args = parser.parse_args()
    # mode = 'traditional'
    main3(args.kind, args.retrieval_k, args.mode, None, args.model_id, args.cat, args.temp, args.shot, args.level)