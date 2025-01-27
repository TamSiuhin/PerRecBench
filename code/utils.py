import re
import json
from rank_bm25 import BM25Okapi
import random
max_retries = 5


def extract_braced_strings(text):
    # Regular expression to find all substrings between { and }, including the braces
    pattern = r'\{[^}]*\}'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)[-1]
    
    return matches

def extract_first_number(text):
    # Regular expression to match float or integer numbers
    number_pattern = r'[-+]?\d*\.\d+|\d+'
    
    # Search for the pattern in the text
    match = re.search(number_pattern, text)
    
    if match:
        # Convert matched string to float or integer
        number_str = match.group()
        return float(number_str)
    
    return None


def rank_rating(float_list):
    # Create a list of tuples where each tuple is (index, value)
    indexed_list = list(enumerate(float_list))
    
    # Sort the list based on the float values in descending order
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    
    # Create a list of ranks with the same length as the original list
    ranks = [0] * len(float_list)
    
    # Assign ranks based on the sorted list
    for rank, (index, value) in enumerate(sorted_list, start=1):
        ranks[index] = rank
    
    return ranks


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


def get_input_profile(path, level):
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

    with open(f'../data/{path}/userid2profile.json') as f:
        userid2profile = json.load(f)

    return items, users, groups, userid2profile

def make_template_point(index, retrieval_num, need_profile=False, rel_rating=False, cot=False):
    if cot:
        with open(f'templates_point/0ours/0ours{index}_cot.txt', 'r') as f:
            template = f.read()
    else:
        with open(f'templates_point/0ours/0ours{index}.txt', 'r') as f:
            template = f.read()
    his = "<|The Start of User Data|>\n"
    if need_profile:
        his += "## User Profile\n{}\n\n"
        his += "## User Most Common Rating\n{:.1f}\n\n"
        his += "## User Average Rating\n{:.1f}\n"

    his += "\n{}\n" * retrieval_num
    his += "<|The End of User Data|>\n"
    template = template.format(his, "{}")

    if rel_rating:
        with open(f'templates_point/0ours/0ours{index}_sys_rel_rating.txt', 'r') as f:
            sys_prompt = f.read()
    elif cot:
        with open(f'templates_point/0ours/0ours{index}_sys_cot.txt', 'r') as f:
            sys_prompt = f.read()
    else:
        with open(f'templates_point/0ours/0ours{index}_sys.txt', 'r') as f:
            sys_prompt = f.read()
    return template, sys_prompt


def make_shot_point(index, behavior_all, history_idx, items, retrieval_num, shot, profile=None, rel_rating=False):
    with open(f'templates_point/0ours/0ours{index}.txt', 'r') as f:
        template = f.read()
    
    corpus = []
    for behavior in behavior_all[:history_idx]:
        history = encode_history(behavior=behavior, item_dict=items, rel_rating=rel_rating)
        corpus.append(history)
    
    if profile is not None:
        avg_rating, common_rating = get_avg_common_rating(behavior_all)

    shot_prompt = ''
    sampled_idx = random.sample(list(range(12, len(corpus))), shot)

    for i in sampled_idx:
        
        sampled_behavior = behavior_all[i]

        retrieved_corpus = corpus[:i]
        retrieved_his = retrieval(retrieved_corpus, corpus[i], retrieval_num)

        his_temp = "<|The Start of User Data|>\n"

        if profile is not None:
            his_temp += "## User Profile\n{}\n\n".format(profile)
            his_temp += "## User Most Common Rating\n{:.1f}\n\n".format(common_rating)
            his_temp += "## User Average Rating\n{:.1f}\n".format(avg_rating)

            # his_temp += "\n## User Profile\n{}\n\n".format(profile)
        
        his_temp += "\n{}\n\n" * len(retrieved_his)
        his_temp += "<|The End of User Data|>\n\n"
        
        his = his_temp.format(*retrieved_his)

        query = encode_query(sampled_behavior['parent_asin'], items)

        template_example = template.format(his, query)
        
        if rel_rating:
            template_example += '\n{{"predicted_rating": {:.1f}}}\n\n'.format(sampled_behavior['relative_rating'])
        else:
            template_example += '\n{{"predicted_rating": {:.1f}}}\n\n'.format(sampled_behavior['rating'])

        # if profile is not None:
        #     template_example = f"[User Profile]\n{profile}\n\n" + template_example

        shot_prompt += template_example

    return shot_prompt

def make_shot_group(index, id2behavior, items, retrieval_num, shot, cat, userid2profile=None, rel_rating=False, level='easy'):
    with open(f'templates_list/0ours/0ours{index}.txt', 'r') as f:
        template = f.read()
    
    with open(f'../data/{cat}/{cat}-processed/held_out-{level}.json', 'r') as f:
        held_out = json.load(f)

    group_list = []
    asin_list = []
    for asin, groups in held_out.items():
        for group in groups:
            group_list.append(group)
            asin_list.append(asin)
    
    selected_idx = random.sample(list(range(len(asin_list))), shot)

    selected_group_list = []
    selected_asin_list = []
    
    for idx in selected_idx:
        selected_group_list.append(group_list[idx])
        selected_asin_list.append(asin_list[idx])
    
    shot_prompt = ''

    for idx in range(shot):
        
        group = selected_group_list[idx]
        query_asin = selected_asin_list[idx]

        his_group = ''
        query = encode_query(query_asin, items)

        relative_rating_list = []

        for user_idx, b in enumerate(group):
            corpus = []
            behavior_all = id2behavior[b['user_id']]
            history_idx = b['history_len']

            for behavior in behavior_all[:history_idx]:
                history = encode_history(behavior, items, user_idx+1, rel_rating)
                corpus.append(history)

            avg_rating, common_rating = get_avg_common_rating(behavior_all)

            retrieved_his = retrieval(corpus, query, retrieval_num)
            if userid2profile is not None:
                profile = userid2profile[b['user_id']]

                his_temp = "<|The Start of User{} Data|>\n## User Profile\n{}\n\n## User Most Common Rating\n{:.1f}\n\n## User Average Rating\n{:.1f}\n\n{}\n<|The End of User{} Data|>\n\n".format(user_idx+1, profile, common_rating, avg_rating, "\n\n".join(["{}"] * len(retrieved_his)), user_idx+1) 
            
            else:
                his_temp = "<|The Start of User{} Data|>\n{}\n<|The End of User{} Data|>\n\n".format(user_idx+1, "\n\n".join(["{}"] * len(retrieved_his)), user_idx+1) 
            his = his_temp.format(*retrieved_his)
            his_group += his

            relative_rating_list.append(b['relative_rating'])

        real_ranking = rank_rating(relative_rating_list)
        template_example = template.format(his_group, query)
        template_example += '\n{{"predicted_ranking": {}}}\n\n'.format(real_ranking)

        shot_prompt += template_example

    return shot_prompt

def make_template_group(index, retrieval_num, user_cnt, need_profile=False, rel_rating=False, cot=False):
    if not cot:
        with open(f'templates_list/0ours/0ours{index}.txt', 'r') as f:
            template = f.read()
    else:
        with open(f'templates_list/0ours/0ours{index}_cot.txt', 'r') as f:
            template = f.read()

    his = ""
    for i in range(user_cnt):

        user_idx = i+1
        his += "<|The Start of User{} Data|>\n".format(user_idx)
        if need_profile:
            his += "## User Profile\n{}\n\n"
            his += "## User Most Common Rating\n{:.1f}\n\n"
            his += "## User Average Rating\n{:.1f}\n"

        his += "\n{}\n".format("{}") * retrieval_num
        his += "<|The End of User{} Data|>\n\n".format(user_idx)

    template = template.format(his, "{}")

    if rel_rating:
        with open(f'templates_list/0ours/0ours{index}_sys_rel_rating.txt', 'r') as f:
            sys_prompt = f.read()
        
    elif cot:
        with open(f'templates_list/0ours/0ours{index}_sys_cot.txt', 'r') as f:
            sys_prompt = f.read()
    else:
        with open(f'templates_list/0ours/0ours{index}_sys.txt', 'r') as f:
            sys_prompt = f.read()
        
    return template, sys_prompt

def find_longest_string(strings):
    # Initialize the longest string as an empty string
    longest = ""

    # Loop through each string in the list
    for string in strings:
        # If the current string is longer than the longest string found so far, update it
        if len(string) > len(longest):
            longest = string

    return longest


def encode_query(id, item_dict):
    item = item_dict[id]

    prompt = "<|The Start of Query Item Information|>\n"
    prompt += "### Item Title\n{}\n".format(item['title'])
    try:
        prompt += "### Item Author\n{}\n".format(item['author']['name'])
    except:
        # print(item['description'])
        pass
    
    # print(item)

    # if len(item['features'])!=0:
    #     prompt += "### Item Feature\n{}\n".format(find_longest_string(item['features']))

    prompt += "<|The End of Query Item Information|>"
    return prompt

def encode_history(behavior, item_dict, idx=None, rel_rating=False):
    item = item_dict[behavior['parent_asin']]

    prompt = ""
    prompt += "### Item Title\n{}\n".format(item['title'])
    try:
        prompt += "### Item Author\n{}\n".format(item['author']['name'])
    except:
        # print(item['description'])
        pass

    # if len(item['features'])!=0:
    #     prompt += "### Item Feature\n{}\n".format(find_longest_string(item['features']))

    # if 'features' in behavior.keys():
    #     bt = behavior['features']
    #     prompt += "### User Review\n{}\n".format(bt)
    if rel_rating == False:
        if idx is not None:
            prompt += "### User{} Rating\n{}".format(idx, behavior['rating'])
        else:
            prompt += "### User Rating\n{}".format(behavior['rating'])
    else:
        # print(behavior['relative_rating'])

        if idx is not None:
            prompt += "### User{} Rating\n{:.1f}".format(idx, behavior['relative_rating'])
        else:
            prompt += "### User Rating\n{:.1f}".format(behavior['relative_rating'])

    return prompt

def retrieval(corpus, query, n):
    tokenized_corpus = [doc.split() for doc in corpus]
    tokenized_query = query.split(" ")
    bm25 = BM25Okapi(tokenized_corpus)
    profiles = bm25.get_top_n(tokenized_query, corpus, n)
    return profiles

def check_format(input, user_cnt):
    input_dict = json.loads(input)
    Flag = True
    if "predicted_ranking" not in input_dict.keys() and "reason" not in input_dict.keys():
        Flag = False
        print('key')
    if len(input_dict["predicted_ranking"]) != user_cnt or min(input_dict["predicted_ranking"])!=1 or max(input_dict["predicted_ranking"])!=len(input_dict["predicted_ranking"]):
        Flag = False
        print('user_cnt')
    # for i in input_dict['predicted_ranking']:
    #     if not bool(re.match(r"^User\d+$", i)):
    #         Flag = False
    #         print('user_name')
            
    if Flag ==False:
        print('wrong format!!')
        raise Exception('wrong format')
    
    return Flag

from collections import Counter

def majority_vote(strings):
    if not strings:
        return None  # Return None if the list is empty

    # Count the occurrences of each string
    counts = Counter(strings)
    
    # Find the string with the highest count
    majority_string, _ = counts.most_common(1)[0]
    
    return majority_string

def avg_prediction(strings):
    score = []
    
    for s in strings:
        out = extract_braced_strings(s)
        out = json.loads(out)
        for k in out.keys():
            if 'rating' in k.lower() or 'score' in k.lower():
                r = k
        pred = out[r]
        score.append(pred)

    return float(np.mean(score))

def heapify(arr, n, i, item, items, id2behavior, userid2profile, held_out, model_id=None, engine='openai', rel_rating=False, temp=0.1, max_tokens=512, shot=0, retrieval_num=4, cat='all', use_profile=False, level='easy', consistency=1, cot=False):
    largest = i  # Initialize largest as root
    left = 2 * i + 1  # left = 2*i + 1
    right = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is greater than root
    if left < n and compare_two_users_pref(arr[left][0], arr[largest][0], item, items, id2behavior, userid2profile, held_out, model_id, engine, rel_rating, temp, max_tokens, shot, retrieval_num, cat, use_profile, level, consistency, cot):
        largest = left

    # See if right child of root exists and is greater than largest so far
    if right < n and compare_two_users_pref(arr[right][0], arr[largest][0], item, items, id2behavior, userid2profile, held_out, model_id, engine, rel_rating, temp, max_tokens, shot, retrieval_num, cat, use_profile, level, consistency, cot):
        largest = right

    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap

        # Heapify the root
        heapify(arr, n, largest, item, items, id2behavior, userid2profile, held_out, model_id, engine, rel_rating, temp, max_tokens, shot, retrieval_num, cat, use_profile, level, consistency, cot)


def heapsort(text_list,
    item=None,
    model_id=None,
    engine='openai', 
    rel_rating=False,
    temp=0.1,
    max_tokens=512,
    shot=0, 
    retrieval_num=4, 
    cat='all', 
    use_profile=False, 
    level='easy',
    items=None, 
    id2behavior=None, 
    userid2profile=None, 
    held_out=None,
    consistency=1, 
    cot=False):

    # Pair each text with its original index
    arr = [(text, i) for i, text in enumerate(text_list)]
    n = len(arr)

    # Build a maxheap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, item, items, id2behavior, userid2profile, held_out, model_id, engine, rel_rating, temp, max_tokens, shot, retrieval_num, cat, use_profile, level, consistency, cot)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0, item, items, id2behavior, userid2profile, held_out, model_id, engine, rel_rating, temp, max_tokens, shot, retrieval_num, cat, use_profile, level, consistency, cot)

    # Separate sorted texts and original indices
    sorted_texts = [text for text, _ in arr][::-1]
    argsort_indices = [index for _, index in arr][::-1]

    return sorted_texts, argsort_indices

import call_api


def extract_user(text):
    # Regular expression to find all substrings between { and }, including the braces
    pattern = r'\[[^}]*]'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    if len(matches) != 0:
        return matches[0]

    if 'user a' or 'user b' in text.lower():
        return text.strip()

def user_template_pair(retrieval_num, need_profile=False):
    his = "<|The Start of User Data|>\n"
    if need_profile:
        his += "## User Profile\n{}\n\n"
        his += "## User Most Common Rating\n{:.1f}\n\n"
        his += "## User Average Rating\n{:.1f}\n"

    his += "\n{}\n\n" * retrieval_num
    his += "<|The End of User Data|>\n"
    template = his

    return his

with open('./template_pair/0ours0.txt', 'r') as f:
    pair_prompt_template = f.read()

def make_shot_pair(retrieval_num, shot, items, id2behavior, userid2profile, held_out, use_profile=False, rel_rating=False, level='easy'):

    # items, id2behavior, _, userid2profile = get_input_profile(f"{cat}/{cat}-processed", level)

    # with open(f'../data/{cat}/{cat}-processed/held_out-{level}.json') as f:
    #     held_out = json.load(f)
    
    group_list = []
    asin_list = []
    for asin, groups in held_out.items():
        for group in groups:
            group_list.append(group)
            asin_list.append(asin)

    selected_idx = random.sample(list(range(len(asin_list))), shot)

    selected_group_list = []
    selected_asin_list = []
    
    for idx in selected_idx:
        selected_group_list.append(group_list[idx])
        selected_asin_list.append(asin_list[idx])

    shot_prompt = ''

    for idx in range(shot):
        
        group = selected_group_list[idx]
        query_asin = selected_asin_list[idx]

        his_group = ''
        query = encode_query(query_asin, items)

        template = user_template_pair(retrieval_num, use_profile)
        user_data_list = []
        user_rel_rating_list = []

        idx_lst = list(np.arange(len(group)))

        for user_idx in random.sample(idx_lst, 2):
            user = group[user_idx]
            behaviors = id2behavior[user['user_id']]

            corpus = []
            retr_user_his = []
            for behavior in behaviors[:user['history_len']]:
                history = encode_history(behavior=behavior, item_dict=items, idx=None, rel_rating=rel_rating)
                corpus.append(history)

            if use_profile:
                avg_rating, common_rating = get_avg_common_rating(behaviors)
                profile = userid2profile[user['user_id']]
                retr_user_his += [profile, common_rating, avg_rating]
            
            retr_user_his += retrieval(corpus, query, retrieval_num)
            user_str = template.format(*retr_user_his)
            user_data_list.append(user_str)
            user_rel_rating_list.append(user['relative_rating'])

        # print(user_data_list)
        if user_rel_rating_list[0]> user_rel_rating_list[1]:
            output = "[User A]"
        else:
            output = "[User B]"
        
        shot_prompt += pair_prompt_template.format(user_data_list[0], user_data_list[1], query)
        
        shot_prompt += "\n" + output + "\n\n"
    return shot_prompt

# Example COMPARE function
def compare_two_users_pref(A: str,
    B: str,
    item: str,
    items: dict, 
    id2behavior: dict,
    userid2profile: dict,
    held_out: dict,
    model_id: str=None,
    engine :str = "openai", 
    rel_rating: bool=False,
    temp=0.1,
    max_tokens=512, 
    shot=0, 
    retrieval_num=4, 
    cat='all',
    use_profile=False,
    level='easy',
    consistency=1,
    cot=False):

    if not rel_rating:
        with open("./template_pair/0ours0_sys.txt", 'r') as f:
            sys_prompt = f.read()
    else:
        with open("./template_pair/0ours0_sys_rel_rating.txt", 'r') as f:
            sys_prompt = f.read()

    if not cot:
        with open('./template_pair/0ours0.txt', 'r') as f:
            prompt_template = f.read()
    else:
        with open('./template_pair/0ours0_cot.txt', 'r') as f:
            prompt_template = f.read()
    
    prompt_1 = prompt_template.format(A, B, item)
    prompt_2 = prompt_template.format(B, A, item)
    
    if shot>0:
        shot_prompt = make_shot_pair(
            retrieval_num, shot, items, id2behavior, userid2profile, held_out, use_profile=False, rel_rating=False, level='easy'
        )
        prompt_1 = shot_prompt + prompt_1
        prompt_2 = shot_prompt + prompt_2

    # print(prompt_1)

    # print("*"*100)
    # print(prompt_2)

    # print("#"*100)

    if engine=="anthropic":
        message_1 = [
            {"role": "user", "content": prompt_1}
        ]
    else:
        message_1 = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_1}
        ]
    
    # print(prompt_1)

    retries = 0
    while retries < max_retries:
        try:
            # print('INPUT')
            output_1 = call_api.GetAnswer(
                model_id=model_id,
                message=message_1,
                sys_prompt=sys_prompt,
                temp=temp,
                max_tokens=max_tokens,
                engine=engine,
                n=consistency
            )
            # print("OUTPUT_1")
            # print(output_1)
            output_1 = majority_vote(output_1)
            pred_1 = extract_user(output_1)
            # print("pred_1: {}".format(pred_1))
            assert pred_1.lower() in ["[user a]", "[user b]", "user a", "user b"]
            break
        except:
                retries += 1
                ...

    if engine=="anthropic":
        message_2 = [
            {"role": "user", "content": prompt_2}
        ]
    else:
        message_2 = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_2}
        ]
    
    retries = 0
    while retries < max_retries:
        try:
            output_2 = call_api.GetAnswer(
                model_id=model_id,
                message=message_2,
                sys_prompt=sys_prompt,
                temp=temp,
                max_tokens=max_tokens,
                engine=engine,
                n=consistency
            )
            output_2 = majority_vote(output_2)
            pred_2 = extract_user(output_2)
            # print("pred_2: {}".format(pred_2))
            assert pred_2.lower() in ["[user a]", "[user b]", "user a", "user b"]
            break
        except:
            retries += 1
            ...

    # print(pred_1)
    # print(pred_2)
    if pred_1.lower() == '[user a]' and pred_2.lower() == '[user b]':
        output = True
    elif pred_1.lower() == 'user a' and pred_2.lower() == 'user b':
        output = True
    else:
        output = False

    return output

from collections import Counter
import numpy as np

def get_avg_common_rating(behavior_list):
    
    rating_list = []
    for b in behavior_list:
        rating_list.append(b['rating'])
    
    avg_rating = np.mean(rating_list)
    most_common_rating = Counter(rating_list).most_common(1)[0][0]

    return avg_rating, most_common_rating


import json

def append_to_jsonl(file_path, data):
    """
    Appends data to a JSONL file. Each dictionary in data will be a separate line in the file.
    
    Parameters:
        file_path (str): Path to the JSONL file.
        data (dict or list of dict): A dictionary or a list of dictionaries to append.
    """
    # Ensure data is in list form for consistent handling
    if isinstance(data, dict):
        data = [data]
    
    # Open the file in append mode
    with open(file_path, 'a') as file:
        for entry in data:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')

import itertools
import random

def sample_combinations(input_list, n):
    """
    Generate all combinations of elements in the input list with the same number of elements,
    sample `n` combinations randomly, and return the sampled combinations.
    
    Args:
        input_list (list): The list of elements to generate combinations from.
        n (int): The number of combinations to sample.
        
    Returns:
        list: A list of `n` sampled combinations.
    """
    if not input_list:
        return []

    # Generate all combinations with the same length as the input list
    all_combinations = list(itertools.permutations(input_list, len(input_list)))
    
    # Sample `n` combinations randomly
    if n > len(all_combinations):
        return all_combinations
    else:
        sampled_combinations = random.sample(all_combinations, n)
        return sampled_combinations

# Example usage:
# input_list = [1, 2, 3]
# n = 3
# print(sample_combinations(input_list, n))


import itertools
import random
from typing import List, Any

def sample_multi_length_combinations(input_list: List[Any], n: int, lengths: List[int]) -> List[List[Any]]:
    """
    Generate combinations of elements in the input list with specified lengths,
    sample `n` combinations for each length, and return all sampled combinations.
    
    Args:
        input_list (List[Any]): The list of elements to generate combinations from.
        n (int): The number of combinations to sample for each length.
        lengths (List[int]): List of lengths for which to generate combinations.
        
    Returns:
        List[List[Any]]: A list containing all sampled combinations of different lengths.
    """
    if not input_list or not lengths:
        return []
    
    # Validate lengths
    max_length = len(input_list)
    valid_lengths = [l for l in lengths if 0 < l <= max_length]
    
    if not valid_lengths:
        raise ValueError(f"No valid lengths provided. Lengths must be between 1 and {max_length}")
    
    all_sampled_combinations = []
    
    for length in valid_lengths:
        # Generate all combinations of current length
        all_combinations = list(itertools.permutations(input_list, length))
        
        # Sample combinations for current length
        if n > len(all_combinations):
            sampled = all_combinations
        else:
            sampled = random.sample(all_combinations, n)
        
        # Convert tuples to lists and add to result
        all_sampled_combinations.extend([list(comb) for comb in sampled])
    
    return all_sampled_combinations
