from datasets import load_dataset, DatasetDict

# for i in ['point', 'group', 'pair']:
#     dataset = load_dataset('json', data_files=f'/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/data/all/train-small-{i}.jsonl')

#     data_dict = DatasetDict({
#         'train_sft': dataset['train']
#     })
#     data_dict.save_to_disk(f'/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/data/all/train_data-small-{i}')

#     print(data_dict['train_sft'])


# from datasets import load_dataset, load_from_disk
# datasets = load_from_disk('/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/data/all/train_data-point')

# print(datasets['train_sft']['messages'][0])


# for i in ["mix"]:
#     dataset = load_dataset('json', data_files=f'/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/data/all/train-{i}.jsonl')

#     data_dict = DatasetDict({
#         'train_sft': dataset['train']
#     })
#     data_dict.save_to_disk(f'/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/data/all/train_data-{i}')

#     print(data_dict['train_sft'])




from datasets import load_dataset, load_from_disk

dataset = load_from_disk('/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/data/all/train_data-mix')
print(dataset['train_sft']['messages'][0])

# import json
# with open('/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/data/all/train-mix.jsonl', 'r') as f:
#     for line in f.readlines():
#         i = json.loads(line)
#         print(i)
#         break