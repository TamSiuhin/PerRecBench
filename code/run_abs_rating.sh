#### OPENAI API
# export OPENAI_API_KEY="YOUR_API_KEY"
# export OPENAI_API_BASE="https://api.openai.com/v1"


#### DEEPSEEK
# export DEEPSEEK_API_KEY="YOUR_API_KEY"
# export OPENAI_API_BASE="https://api.deepseek.com/v1"

### VLLM API
export OPENAI_API_BASE="http://localhost:8002/v1"
export OPENAI_API_KEY="EMPTY"


#### TOGETHER API
# export TOGETHER_API_KEY="YOUR_API_KEY"


#### AIML API
# export OPENAI_API_BASE="https://api.aimlapi.com/v1"
# export OPENAI_API_KEY="YOUR_API_KEY"

#### ANTHROPIC API
# export ANTHROPIC_API_KEY="YOUR_API_KEY"


#### MISTRAL API
# export MISTRAL_API_KEY="YOUR_API_KEY"


# MODEL_ID="Mistral-Nemo-Instruct-2407"
# MODEL_ID='gpt-4o-mini-2024-07-18'
MODEL_ID='Mistral-7B-Instruct-v0.3'
k=4
# level='medium'
# mode='group'
temp=0.1
shot=0
engine="openai" # can be chosen from ['openai', 'mistral', 'anthropic', 'together', 'google']

# # 0 shot - point
# python main_model_fewshot-profile.py --model_id $MODEL_ID --n 1 --temp $temp --retrieval_k $k --shot $shot --engine openai --mode point --level $level

# # 0 shot group
# python main_model_fewshot-profile.py --model_id $MODEL_ID --n 1 --temp $temp --retrieval_k $k --shot $shot --engine openai --mode group --level $level

# # 0 shot pair
# python main_model_fewshot-profile.py --model_id $MODEL_ID --n 1 --temp $temp --retrieval_k $k --shot $shot --engine openai --mode pair --level $level


########################################################################################################
# with profile (DEFAULT)

for level in "easy" "medium" "hard"; do
    python main_model_fewshot-profile.py --model_id $MODEL_ID --n 1 --temp $temp --retrieval_k $k --shot $shot --engine $engine --mode point --level $level --add_profile 
    python main_model_fewshot-profile.py --model_id $MODEL_ID --n 1 --temp $temp --retrieval_k $k --shot $shot --engine $engine --mode group --level $level --add_profile
    python main_model_fewshot-profile.py --model_id $MODEL_ID --n 1 --temp $temp --retrieval_k $k --shot $shot --engine $engine --mode pair --level $level --add_profile
done

########################################################################################################
# with CoT

# for level in "easy" "medium" "hard"; do
#     python main_model_fewshot-profile.py --model_id $MODEL_ID --n 1 --temp $temp --retrieval_k $k --shot $shot --engine $engine --mode point --level $level --add_profile --cot
#     python main_model_fewshot-profile.py --model_id $MODEL_ID --n 1 --temp $temp --retrieval_k $k --shot $shot --engine $engine --mode group --level $level --add_profile --cot
#     python main_model_fewshot-profile.py --model_id $MODEL_ID --n 1 --temp $temp --retrieval_k $k --shot $shot --engine $engine --mode pair --level $level --add_profile --cot
# done

########################################################################################################
# with self-consistency

# consistency=5
# for level in "easy" "medium" "hard"; do
#     python main_model_fewshot-profile.py --model_id $MODEL_ID --n $consistency --temp $temp --retrieval_k $k --shot $shot --engine $engine --mode point --level $level --add_profile
#     python main_model_fewshot-profile.py --model_id $MODEL_ID --n $consistency --temp $temp --retrieval_k $k --shot $shot --engine $engine --mode group --level $level --add_profile
#     python main_model_fewshot-profile.py --model_id $MODEL_ID --n $consistency --temp $temp --retrieval_k $k --shot $shot --engine $engine --mode pair --level $level --add_profile
# done