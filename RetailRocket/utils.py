import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import dok_matrix
import pandas as pd


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) < length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


def set_device():
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def prepare_dataloader(data_path, batch_size):
    replay_buffer = pd.read_pickle(data_path + 'replay_buffer.df')
    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
    actions = replay_buffer_dic['action'].values()
    actions = torch.from_numpy(np.fromiter(actions, dtype=np.long)).long()
    next_states = replay_buffer_dic['next_state'].values()
    next_states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in next_states]).long()
    len_next_states = replay_buffer_dic['len_next_states'].values()
    len_next_states = torch.from_numpy(np.fromiter(len_next_states, dtype=np.long)).long()
    is_buy = replay_buffer_dic['is_buy'].values()
    is_buy = torch.from_numpy(np.fromiter(is_buy, dtype=np.long)).long()
    is_done = replay_buffer_dic['is_done'].values()
    is_done = torch.from_numpy(np.fromiter(is_done, dtype=np.bool))
    train_data = TensorDataset(states, len_states, actions, next_states,
                               len_next_states, is_buy, is_done)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def get_one_hot_item_sess(data_path):
    sorted_events = pd.read_csv(data_path + 'sorted_events.csv')
    item_sess_one_hot = dok_matrix(
        shape=(sorted_events.item_id.max() + 1, sorted_events.session_id.max() + 1),
        dtype=np.int32
    )
    for item_id, session_id in zip(sorted_events.item_id.values, sorted_events.session_id.values):
        item_sess_one_hot[item_id, session_id] = 1
    return item_sess_one_hot


def calculate_hit(sorted_list, topk, true_items, rewards, r_click,
                  total_reward, hit_click, ndcg_click, hit_purchase, ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def get_stats(data_path):
    data_statis = pd.read_pickle(data_path + 'data_statis.df')  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    return state_size, item_num

