import json
from rank_bm25 import BM25Okapi
# from openai import OpenAI
import os
from tqdm import tqdm
import random
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau
import sys
import concurrent.futures
import re
from utils import rank_rating, get_input_profile, make_template_group, make_shot_group, find_longest_string, extract_braced_strings, make_template_point, encode_query, encode_history, retrieval, check_format, majority_vote, make_shot_point
from utils import heapsort, compare_two_users_pref, user_template_pair, compare_two_users_pref, get_avg_common_rating, get_input, avg_prediction
import call_api
import numpy as np

max_retries = 5

def main0_few_shot_profile(index, rn, mode, model_id, cat, temp=0.6, shot=0, use_profile=False, engine="openai", n=1, rel_rating=False, level='easy', cot=False):

    retrieval_num = rn
    if "processed" in mode:
        items, users, groups, userid2profile = get_input_profile(f"{cat}/{cat}-{mode.split('-')[0]}", level)
    else:
        items, users, groups = get_input(f"{cat}/{cat}-{mode.split('-')[0]}", level)

    if mode == 'processed-point':
        template, system_prompt = make_template_point(index, retrieval_num, use_profile, rel_rating, cot)
        output = {}
        input = {}
        result = {}
        for key in tqdm(groups.keys()):
            output[key] = []
            input[key] = []
            result[key] = []
            q = encode_query(key, items)
            for group in groups[key]:
                input[key].append([])
                output[key].append([])
                result[key].append([])
                for user in group:
                    behaviors = users[user['user_id']]
                    avg_rating, common_rating = get_avg_common_rating(behaviors)

                    corpus = []
                    for behavior in behaviors[:user['history_len']]:
                        history = encode_history(behavior=behavior, item_dict=items, idx=None, rel_rating=rel_rating)
                        corpus.append(history)
                    retr_his = retrieval(corpus, q, retrieval_num)
                    if use_profile:
                        profile = userid2profile[user['user_id']]
                        retr_his = [profile, common_rating, avg_rating] + retr_his
                        
                    prompt = template.format(*retr_his, q)
                    
                    if shot !=0:
                        if use_profile:
                            prompt_shot = make_shot_point(index, behaviors, user['history_len'], items, retrieval_num, shot, profile, rel_rating)
                        else:
                            prompt_shot = make_shot_point(index, behaviors, user['history_len'], items, retrieval_num, shot, None, rel_rating)
                        prompt = prompt_shot + prompt
                        
                    # print(prompt)
                    # print('-------------------------------------')
                    # prompt += '\nAnswer:'

                    retries = 0
                    correct_flag = False
                    while retries < max_retries:
                        if engine=="anthropic":
                            message = [
                                {"role": "user", "content": prompt}
                            ]
                        else:
                            message = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ]
                        
                        out = call_api.GetAnswer(
                            model_id=model_id, 
                            engine=engine,
                            temp=temp, 
                            message=message,
                            sys_prompt=system_prompt,
                            n=n
                            )
                        try:
                            out = majority_vote(out)
                            out = extract_braced_strings(out)
                            out = json.loads(out)
                            print(out)
                            print('-------------------------------------')

                            for k in out.keys():
                                if 'rating' in k.lower() or 'score' in k.lower():
                                    r = k
                            pred = out[r]
                            # pred = avg_prediction(out)

                            if isinstance(pred, int) or isinstance(pred, float):
                                result[key][-1].append(pred)

                                output[key][-1].append(out)
                                input[key][-1].append(prompt)
                                correct_flag = True

                                break
                        except:
                            print('something wrong')
                            retries += 1
                            ...
                    
                    if correct_flag == False:
                        print("format not correct")
                        pred = "format_not_correct" #random.randint(1, 5)
                        result[key][-1].append(pred)

                        output[key][-1].append({"predicted_rating": pred})
                        input[key][-1].append(prompt)

    elif mode == 'processed-group':
        output = {}
        input = {}
        result = {}

        for key in tqdm(groups.keys()):
            output[key] = []
            input[key] = []
            result[key] = []
            q = encode_query(key, items)

            for group in groups[key]:
                input[key].append([])
                output[key].append([])
                result[key].append([])
                
                template, system_prompt = make_template_group(index, retrieval_num, len(group), use_profile, rel_rating, cot)

                retr_his = []
                for user_idx, user in enumerate(group):
                    behaviors = users[user['user_id']]
                    corpus = []
                    for behavior in behaviors[:user['history_len']]:
                        history = encode_history(behavior=behavior, item_dict=items, idx=user_idx+1, rel_rating=rel_rating)
                        corpus.append(history)
    
                    if use_profile:
                        avg_rating, common_rating = get_avg_common_rating(behaviors)

                        profile = userid2profile[user['user_id']]
                        # retr_his.append(profile)
                        retr_his += [profile, common_rating, avg_rating]

                    retr_his += retrieval(corpus, q, retrieval_num)

                prompt = template.format(*retr_his, q)

                if shot != 0:
                    if use_profile:
                        shot_prompt = make_shot_group(index, users, items, retrieval_num, shot, cat, userid2profile, rel_rating, level)
                    else:
                        shot_prompt = make_shot_group(index, users, items, retrieval_num, shot, cat, None, rel_rating, level)
                    prompt = shot_prompt + prompt
                
                retries = 0
                correct_flag = False
                
                while retries < max_retries:
                    if engine=="anthropic":
                        message = [
                            {"role": "user", "content": prompt}
                        ]
                    else:
                        message = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]

                    
                    out = call_api.GetAnswer(
                            model_id=model_id, 
                            engine=engine,
                            temp=temp, 
                            message=message,
                            sys_prompt=system_prompt,
                            n=n
                        )

                    try:  
                        out = majority_vote(out)
                        out = extract_braced_strings(out)
                        print(out)
                        check_format(out, len(group))
                        print('format correct')
                        print('----------------------------------')
                        out = json.loads(out)
                        
                        for k in out.keys():
                            if 'rank' in k:
                                r = k
                        result[key][-1].append(out[r])

                        output[key][-1].append(out)

                        input[key][-1].append(prompt)
                        correct_flag=True
                        break
                    except:
                        print('something went wrong!')
                        retries += 1
                        ...

                if correct_flag == False:
                    print("use random prediction")
                    pred = "format_not_correct" # list(range(1, len(group)+1))
                    # random.shuffle(pred)
                    
                    result[key][-1].append(pred)

                    output[key][-1].append({"predicted_ranking": pred})
                    input[key][-1].append(prompt)
    
    elif mode == 'processed-pair':
        output = {}
        input = {}
        result = {}

        items, id2behavior, _, userid2profile = get_input_profile(f"{cat}/{cat}-processed", level)
        
        with open(f'../data/{cat}/{cat}-processed/held_out-{level}.json') as f:
            held_out = json.load(f)

        for key in tqdm(groups.keys()):
            output[key] = []
            input[key] = []
            result[key] = []
            q = encode_query(key, items)

            for group in groups[key]:
                input[key].append([])
                output[key].append([])
                result[key].append([])
                
                template = user_template_pair(retrieval_num, use_profile)

                user_data_list = []
                
                for user_idx, user in enumerate(group):
                    behaviors = users[user['user_id']]
                    corpus = []
                    retr_user_his = []
                    for behavior in behaviors[:user['history_len']]:
                        history = encode_history(behavior=behavior, item_dict=items, idx=None, rel_rating=rel_rating)
                        corpus.append(history)
    
                    if use_profile:
                        avg_rating, common_rating = get_avg_common_rating(behaviors)
                        profile = userid2profile[user['user_id']]
                        retr_user_his += [profile, common_rating, avg_rating]

                    retr_user_his += retrieval(corpus, q, retrieval_num)
                    user_str = template.format(*retr_user_his)
                    user_data_list.append(user_str)
                
                # print(user_data_list)
                # print(len(user_data_list))


                sorted_user_list, user_idx_list = heapsort(user_data_list, q, model_id, engine, rel_rating, temp, 512, shot, retrieval_num, cat, use_profile, level, items, id2behavior, userid2profile, held_out, n, cot)
                pred = (np.array(user_idx_list) + 1).tolist()

                print("PRED_RANKING: {}".format(pred))
                
                result[key][-1].append(pred)
                output[key][-1].append({"predicted_ranking": pred})
                input[key][-1].append("pairwise prompt")
                
                # if shot != 0:
                #     if use_profile:
                #         shot_prompt = make_shot_group(index, users, items, retrieval_num, shot, cat, userid2profile, rel_rating)
                #     else:
                #         shot_prompt = make_shot_group(index, users, items, retrieval_num, shot, cat, None, rel_rating)
                #     prompt = shot_prompt + prompt
                
                # retries = 0
                # correct_flag = False
                
                # while retries < max_retries:
                #     if engine=="anthropic":
                #         message = [
                #             {"role": "user", "content": prompt}
                #         ]
                #     else:
                #         message = [
                #             {"role": "system", "content": system_prompt},
                #             {"role": "user", "content": prompt}
                #         ]

                #     out = call_api.GetAnswer(
                #         model_id=model_id, 
                #         engine=engine,
                #         temp=temp, 
                #         message=message,
                #         sys_prompt=system_prompt,
                #         n=n
                #     )
                    
                #     out = majority_vote(out)

                #     try: 
                #         out = extract_braced_strings(out)
                #         print(out)
                #         check_format(out, len(group))
                #         print('format correct')
                #         print('----------------------------------')
                #         out = json.loads(out)
                        
                #         for k in out.keys():
                #             if 'rank' in k:
                #                 r = k
                #         result[key][-1].append(out[r])

                #         output[key][-1].append(out)

                #         input[key][-1].append(prompt)
                #         correct_flag=True
                #         break
                #     except:
                #         print('something went wrong!')
                #         retries += 1
                #         ...

                # if correct_flag == False:
                #     print("use random prediction")
                #     pred = "format_not_correct" # list(range(1, len(group)+1))
                                        
                #     result[key][-1].append(pred)

                #     output[key][-1].append({"predicted_ranking": pred})
                #     input[key][-1].append(prompt)


    elif mode == 'traditional':
        output = []
        input = []
        result = []
        template, system_prompt = make_template_point(index, retrieval_num, cot)

        for behavior in tqdm(groups):
            key = behavior["parent_asin"]
            q = encode_query(key, items)
            
            behaviors = users[behavior['user_id']]
            corpus = []
            for b in behaviors[:behavior['history_len']]:
                history = encode_history(behavior=b, item_dict=items)
                corpus.append(history)
            retr_his = retrieval(corpus, q, retrieval_num)
            prompt = template.format(*retr_his, q)
            # prompt += '\nAnswer:'
            
            if shot != 0:
                prompt_shot = make_shot_point(index, behaviors, behavior['history_len'], items, retrieval_num, shot)
                prompt = prompt_shot + prompt
            
            retries = 0
            correct_flag = False
            while retries < max_retries:
                if engine=="anthropic":
                    message = [
                        {"role": "user", "content": prompt}
                    ]
                else:
                    message = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                out = call_api.GetAnswer(
                    model_id=model_id, 
                    engine=engine,
                    temp=temp, 
                    message=message,
                    sys_prompt=system_prompt,
                    n=n
                    )
                out = majority_vote(out)

                try:
                    out = extract_braced_strings(out)
                    out = json.loads(out)
                    output.append(out)
                    input.append(prompt)
                    for k in out.keys():
                        if 'rating' in k:
                            r = k
                    result.append(out[r])
                    
                    correct_flag=True
                    break
            
                except:
                    retries += 1
                    ...

            if correct_flag == False:
                print("use random prediction")
                # pred =random.randint(1, 5)
                pred = "format_not_correct"
                output.append({"predicted_rating": pred})
                result.append(pred)
                input.append(prompt)
    
    exp_name = "exp0"
    if use_profile:
        exp_name += "-profile"

    if rel_rating:
        result_dir = "rel_rating"
    else:
        result_dir = 'abs_rating'
        
    directory_in = f"result-{level}/{result_dir}/{model_id}/{shot}shot/{cat}/{exp_name}/k{retrieval_num}/{mode}/input/"

    if not os.path.exists(directory_in):
        # If it doesn't exist, create the directory
        os.makedirs(directory_in)

    directory_out = f"result-{level}/{result_dir}/{model_id}/{shot}shot/{cat}/{exp_name}/k{retrieval_num}/{mode}/output/"

    if not os.path.exists(directory_out):
        # If it doesn't exist, create the directory
        os.makedirs(directory_out)

    

    if not cot:
        with open(os.path.join(directory_in, f"{index}-sc{n}.json"), 'w') as f:
            json.dump(input, f)

        with open(os.path.join(directory_out, f"{index}_original-sc{n}.json"), 'w') as f:
            json.dump(output, f)

        with open(os.path.join(directory_out, f"{index}-sc{n}.json"), 'w') as f:
            json.dump(result, f)
    else:
        with open(os.path.join(directory_in, f"{index}-sc{n}-cot.json"), 'w') as f:
            json.dump(input, f)

        with open(os.path.join(directory_out, f"{index}_original-sc{n}-cot.json"), 'w') as f:
            json.dump(output, f)

        with open(os.path.join(directory_out, f"{index}-sc{n}-cot.json"), 'w') as f:
            json.dump(result, f)