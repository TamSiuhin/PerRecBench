
#  "clothing" "electronic" "kitchen" "group"

MODEL_ID='deepseek-chat'
k=4
# level='medium'
rating="abs_rating"
shot=0
n=1

echo $MODEL_ID
echo "${shot} shot"

for level in "easy" "medium" "hard"; do
    for cat in "all"; do
        for mode in "point" "pair" "group"; do
            python evaluate_single.py \
            --pred_path ./result-$level/$rating/$MODEL_ID/${shot}shot/$cat/exp0-profile/k$k/processed-$mode/output/0-sc$n.json \
            --label_path ../data/$cat/$cat-processed/item2group_avg_diff_sample-$level.json \
            --mode $mode
            echo "$MODEL_ID + $cat + $mode + shot$k + k$k + profile + $level" 
        done
    done
done