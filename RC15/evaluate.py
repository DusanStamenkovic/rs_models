import time
from utils import *
import pandas as pd


def evaluate(model, args, val_or_test, state_size, item_num, device):
    start_time = time.time()
    topk = [5, 10, 20]
    reward_click = args.r_click
    reward_buy = args.r_buy
    if val_or_test == "val":
        eval_sessions = pd.read_pickle(args.data_path + 'sampled_val.df')
    else:
        eval_sessions = pd.read_pickle(args.data_path + 'sampled_test.df')
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated = 0
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0]
    hit_clicks = [0, 0, 0]
    ndcg_clicks = [0, 0, 0]
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    model.to(device)
    model.eval()
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            if evaluated == len(eval_ids):
                break
            id = eval_ids[evaluated]
            group = groups.get_group(id)
            history = []
            for index, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = pad_history(state, state_size, item_num)
                states.append(state)
                action = row['item_id']
                is_buy = row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy == 1:
                    total_purchase += 1.0
                else:
                    total_clicks += 1.0
                actions.append(action)
                rewards.append(reward)
                history.append(row['item_id'])
            evaluated += 1
        states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
        len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
        preds = model.forward_eval(states.to(device).long(), len_states.to(device).long())
        sorted_list = np.argsort(preds.tolist())
        torch.cuda.empty_cache()

        # Evaluate accuracy measurements (NDCGs & HRs)
        calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward,
                      hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)

        del states
        del len_states

    val_acc = 0
    for i in range(len(topk)):
        hr_click = hit_clicks[i] / total_clicks
        hr_purchase = hit_purchase[i] / total_purchase
        ng_click = ndcg_clicks[i] / total_clicks
        ng_purchase = ndcg_purchase[i] / total_purchase
        val_acc = val_acc + hr_click + hr_purchase + ng_click + ng_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
        print('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
        print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    print('#########################################################################')
    print('total time needed for the evaluation : ', time.time() - start_time)
    print('#########################################################################')
    return val_acc
