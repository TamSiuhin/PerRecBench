import numpy as np
from scipy import stats
import json
import argparse

parser = argparse.ArgumentParser(description="Parser for LoRA")
parser.add_argument('--pred_path', type=str, default=None)
# parser.add_argument('--group_path', type=str, default=None)
parser.add_argument('--label_path', type=str, default=None)
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--rel_rating', action="store_true")

args = parser.parse_args()
print('#'*100)

error1 = 0
error2 = 0

def load_label(args):
    with open(args.label_path, 'r') as f:
        data = json.load(f)

    label = {}
    for key in data.keys():
        label[key] = []
        for i in range(len(data[key][0])):
            label[key].append(data[key][0][i]['relative_rating'])
    
    label_rank = {}
    
    for key in data.keys():
        # print("raw_score: {}".format(label[key]))
        
        label_rank[key] = list(np.argsort(label[key]) + 1)[::-1]
        # print("ranking: {}".format(list(np.argsort(label[key]) + 1)[::-1]))
    # print("[label]\n")
    # print(label)
    # print('\n')
    
    return label, label_rank

def load_pred(args):
    # print("[MODE] " + args.mode)
    with open(args.pred_path, 'r') as f:
        pred = json.load(f)
    
    pred_rank = {}

    if args.mode == "group" or args.mode=='pair':
        for k in pred.keys():
            pred[k] = pred[k][0][0]
            # for i in range(len(pred[k])):
                # try:
                # pred[k][i] = int(pred[k][i][-1])
                # except:
                #     global error1
                #     error1 += 1
                #     pred[k] = [i for i in range(1, len(pred[k])+1)]
                #     break
    elif args.mode == "point":
        with open(args.label_path) as f:
            data = json.load(f)
            
        for k in pred.keys():
            pred[k] = pred[k][0]

            if not args.rel_rating:
                for i in range(len(pred[k])):
                    pred[k][i] = pred[k][i] - data[k][0][i]['avg_rating']
        
        for key in pred.keys():
            pred_rank[key] = list(np.argsort(pred[key]) + 1)[::-1]
            # print("raw_score: {}".format(pred[key]))
            # print("rank: {}".format(list(np.argsort(pred[key]) + 1)[::-1]))

    return pred, pred_rank

def compute_tau(pred, label):
    all_tau = []
    for k in pred.keys():
        y_true = label[k]
        y_pred = pred[k]
        # print("LABEL: {}".format(y_true))
        # print("PRED: {}".format(y_pred))
        try:
            # print('--'*10)
            # print("y_pred: {}".format(y_pred))
            # print("y_true: {}".format(y_true))
            tau, _ = stats.kendalltau(y_pred, y_true)
            # print(tau)
            all_tau.append(tau)
        except:
            global error2
            error2 += 1
            continue
    return sum(all_tau) / len(all_tau)

def compute_pearson(pred, label):
    all_pearson = []
    for k in pred.keys():
        y_true = label[k]
        y_pred = pred[k]
        # try:
        # print('--'*10)
        # print("y_pred: {}".format(y_pred))
        # print("y_true: {}".format(y_true))
        
        pearson, _ = stats.pearsonr(y_pred, y_true)
        # print("pearsonr: {}".format(pearson))
        all_pearson.append(pearson)
        # except:
        #     global error2
        #     error2 += 1
        #     continue
    return sum(all_pearson) / len(all_pearson)

from sklearn.metrics import accuracy_score, f1_score

def compute_acc(pred, label):
    correct_list = []
    for k in pred.keys():
        y_true = label[k]
        y_pred = pred[k]
        # try:

        # print("y_pred: {}".format(y_pred))
        # print("y_true: {}".format(y_true))
        
        if y_true == y_pred:
            correct_list.append(1)
        else:
            correct_list.append(0)
        # except:
        #     global error2
        #     error2 += 1
        #     continue
    accuracy = accuracy_score(correct_list, [1] * len(correct_list))

    # F1-score calculation
    f1 = f1_score(correct_list, [1] * len(correct_list))
    
    return accuracy, f1

def matrix(args):
    # rows = ["meta.llama3-8b-instruct-v1_0", "mistral.mixtral-8x7b-instruct-v0_1", "meta.llama3-70b-instruct-v1_0", "cohere.command-r-v1_0", "cohere.command-r-plus-v1_0", "anthropic.claude-3-haiku-20240307-v1_0", "anthropic.claude-3-sonnet-20240229-v1_0", "mistral.mistral-large-2402-v1_0", "gpt-3.5-turbo-0125", "gpt-4o-2024-05-13"]
    # colums = ['book', 'clothing', 'kitchen', 'electronic']
    # mode = ['point', 'group']
    # m = np.zeros((len(rows), len(colums) * len(mode)))
    # for i in range(len(rows)):
        # for j in range(len(colums)):
            # for k in range(len(mode)):
    label, label_rank = load_label(args)
    pred, pred_rank = load_pred(args)
    # global error2
    # before = error2
    # m[i][j * 2 + k] = compute_tau(pred, label)
    # print(label)
    # print(label_rank)
    pearsonr=None

    if args.mode=='point':
        result = compute_tau(pred_rank, label_rank)
        # result = compute_tau(pred, label)

        pearsonr = compute_pearson(pred, label)
        acc, f1 = compute_acc(pred_rank, label_rank)

    elif args.mode=='group' or args.mode=='pair':
        result = compute_tau(pred, label_rank)
        pearsonr=None
        # pearsonr = compute_pearson(pred, label)
        acc, f1 = compute_acc(pred, label_rank)

    return result, pearsonr, acc, f1

m, p, acc, f1 = matrix(args)
# np.savetxt('main_tau.csv', m, delimiter=', ')
print(error1, error2)

print("PRED: {}".format(args.pred_path))
print("LABEL: {}".format(args.label_path))
print("GROUP: {}".format(args.label_path))
print("Kendall-Tau: {}".format(m))
print("Pearsonr: {}".format(p))
print("ACC: {} | F1: {}".format(acc, f1))
