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

indices = [
           ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8."],
            ['- Book A:', '- Book B:', '- Book C:', '- Book D:', '- Book E:', '- Book F:', '- Book G:', '- Book H:'], 
           ["i.", "ii.", "iii.", "iv.", "v.", "vi.", "vii.", "viii"], 
           ["*", "*", "*", "*", "*", "*", "*", "*"], 
           ["#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8"]]

def make_template(index, rn):
    with open(f'templates_point/2liu/2liu{index}.txt', 'r') as f:
        template = f.read()
    if isinstance(index, str):
        index = int(index[0])

    histories = "".join([f"{indices[index][i]} {{}}\n" for i in range(rn)])
    template = template.format(histories, '{}')
    return template

def encode_query(id, item_dict):
    item = item_dict[id]
    prompt = "{}, {}"
    try:
        name = item['author']['name']
    except:
        name = ''
    prompt = prompt.format(item['title'], name)
    return prompt

def encode_history(behavior, item_dict):
    item = item_dict[behavior['parent_asin']]
    prompt = "{}, {}, {}"
    try:
        name = item['author']['name']
    except:
        name = ''
    prompt = prompt.format(item['title'], name, behavior['rating'])
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
#         messages=[{"role": "system", "content": "You are a book recommender system now and designed to output according to the format."},
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

def main2(index, rn, mode, count, model_id, cat, temp, level):
    system_prompt = 'You are a recommender system now and designed to output according to the format.'
    retrieval_num = rn
    items, users, groups = get_input(f"{cat}/{cat}-{mode.split('-')[0]}", level)
    template = make_template(index, retrieval_num)

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
                    for behavior in behaviors[:user['history_len']]:
                        history = encode_history(behavior, items)
                        corpus.append(history)
                    retr_his = retrieval(corpus, q, retrieval_num)
                    prompt = template.format(*retr_his, q)
                    
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
                            # e = out.find('star')
                            # s = e
                            # while s >= 0:
                            #     if out[s] == '(':
                            #         break
                            #     s -= 1
                            # out = int(float(out[s+1:e-1]))
                            out = extract_first_number(out)
                            print(out)
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
            for b in behaviors[:behavior['history_len']]:
                history = encode_history(b, items)
                corpus.append(history)
            retr_his = retrieval(corpus, q, retrieval_num)
            prompt = template.format(*retr_his, q)

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
                    # e = out.find('star')
                    # s = e
                    # while s >= 0:
                    #     if out[s] == '(':
                    #         break
                    #     s -= 1
                    # out = int(float(out[s+1:e-1]))

                    out = extract_first_number(out)
                    print(out)
                    break
                except:
                    ...
            output.append(out)
            input.append(prompt)

    # with open(f"result/{retrieval_num}/{mode}/input/{count}.json", 'w') as f:
    #     json.dump(input, f)
    # with open(f"result/{retrieval_num}/{mode}/output/{count}.json", 'w') as f:
    #     json.dump(output, f)

    directory_in = f"result-correlation/{model_id}/0shot/{cat}/exp2/k{retrieval_num}/{mode}/input/"
    if not os.path.exists(directory_in):
        # If it doesn't exist, create the directory
        os.makedirs(directory_in)

    directory_out = f"result-correlation/{model_id}/0shot/{cat}/exp2/k{retrieval_num}/{mode}/output/"
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
    parser.add_argument('--mode', type=str, default='traditional')
    parser.add_argument('--retrieval_k', type=int, default=4)
    parser.add_argument('--level', type=str, default='easy')
    args = parser.parse_args()
    # mode = 'traditional'
    main2(args.kind, args.retrieval_k, args.mode, None, args.model_id, args.cat, args.temp, args.level)