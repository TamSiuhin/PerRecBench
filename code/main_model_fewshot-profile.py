from exp0_few_shot_profile import main0_few_shot_profile
import json
import argparse
import concurrent.futures

'''
anthropic.claude-3-sonnet-20240229-v1:0
mistral.mistral-large-2402-v1:0
'''

parser = argparse.ArgumentParser(description="Parser for LoRA")
parser.add_argument('--model_id', type=str, default='anthropic.claude-3-sonnet-20240229-v1:0')
parser.add_argument('--kind', type=int, default=0)
parser.add_argument('--cat', type=str, default='all')
parser.add_argument('--temp', type=float, default=0.6)
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--retrieval_k', type=int, default=4)
parser.add_argument('--engine', type=str, default='openai')
parser.add_argument('--add_profile', action="store_true")
parser.add_argument('--mode', type=str, default='group')
parser.add_argument('--rel_rating', action="store_true")
parser.add_argument('--level', type=str, default='easy')
parser.add_argument('--cot', action="store_true")


args = parser.parse_args()

def main(model_id, index, rn, categ, shot, use_profile, engine, n, rel_rating, level, cot):
    print("MODE: {}".format(args.mode))
    print("MODEL: {}".format(model_id))
    print("CATEGORY: {}".format(categ))
    print("LEVEL: {}".format(level))
    print("RETRIEVAL K: {}".format(rn))
    print("SHOT: {}".format(shot))
    print("PROFILE: {}".format(use_profile))
    print("RELATIVE RATING: {}".format(rel_rating))
    
    if use_profile:
        print("USE PROFILE")

    if args.mode=='point':
        main0_few_shot_profile(
            index=index,
            rn=rn, 
            mode='processed-point', 
            model_id=model_id, 
            cat=categ, 
            temp=args.temp, 
            shot=shot, 
            use_profile=use_profile, 
            engine=engine,
            n=n,
            rel_rating=rel_rating,
            level=level,
            cot=cot)
    elif args.mode=='group':
        main0_few_shot_profile(index=index,
            rn=rn, 
            mode='processed-group', 
            model_id=model_id, 
            cat=categ, 
            temp=args.temp, 
            shot=shot, 
            use_profile=use_profile, 
            engine=engine,
            n=n,
            rel_rating=rel_rating,
            level=level,
            cot=cot)
    elif args.mode=='pair':
        main0_few_shot_profile(index=index,
            rn=rn, 
            mode='processed-pair', 
            model_id=model_id, 
            cat=categ, 
            temp=args.temp, 
            shot=shot, 
            use_profile=use_profile, 
            engine=engine,
            n=n,
            rel_rating=rel_rating,
            level=level,
            cot=cot)

    elif args.mode=='traditional':
        main0_few_shot_profile(index=index,
            rn=rn, 
            mode='traditional', 
            model_id=model_id, 
            cat=categ, 
            temp=args.temp, 
            shot=shot, 
            use_profile=use_profile, 
            engine=engine,
            n=n,
            rel_rating=rel_rating,
            level=level,
            cot=cot)
    else:
        print("MODE NOT SUPPORTED!!!")

    
        
if __name__ == '__main__':
    # model_cand = ['mistral.mistral-large-2407-v1:0', 'mistral.mistral-small-2402-v1:0', 'mistral.mistral-7b-instruct-v0:2', 'anthropic.claude-3-haiku-20240307-v1:0', 'meta.llama3-70b-instruct-v1:0', 'anthropic.claude-3-sonnet-20240229-v1:0', 'cohere.command-r-plus-v1:0', 'mistral.mistral-large-2402-v1:0', 'meta.llama3-8b-instruct-v1:0', 'mistral.mixtral-8x7b-instruct-v0:1']
    # model_cand = ['meta.llama2-13b-chat-v1']
    # model_cand = ['gpt-3.5-turbo-0125', 'gpt-4o-mini']

    cat_cand = ['all'] 
    # 'electronic', 'kitchen', 'clothing'
    # cat_cand = ['electronic', 'kitchen', 'clothing']
    cat = args.cat
    id = args.model_id
    rn = args.retrieval_k
    idx = args.kind
    # for cat in cat_cand:
        # for s in [0, 1, 2, 3]:
    # print("shot: {}".format(args.shot))
        # for id in model_cand:
            # print(id)
    main(id, idx, rn, cat, args.shot, args.add_profile, args.engine, args.n, args.rel_rating, args.level, args.cot)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
    #     executor.map(main(args.model_id))

        # if (args.model_company=="anthropic"):
        #     executor.map(call_claude, dataset)
        # elif (args.model_company=="openai"):
        #     executor.map(call_openai, dataset)
    