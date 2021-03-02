import torch
from evaluate import evaluate
from utils import get_stats, set_device
from Caser import Caser
from GRU import GRU
from NextItNet import NextItNet
from SASRec import SASRec

models_path = '../models/'
model_name = ' '  # pick one from {caser, gru, nextitnet , sasrec}


class Args:
    r_click = 0.2
    r_buy = 1.0
    data_path = './data/'
    dilations = '[1, 2, 1, 2, 1, 2]'
    filter_sizes = '[2, 3, 4]'
    hidden_factor = 64
    num_filters = 16
    dropout_rate = 0.1


if __name__ == '__main__':
    args = Args

    device = set_device()
    print('Using {} for training'.format(device))

    state_size, item_num = get_stats(args.data_path)
    assert model_name in {'caser', 'gru', 'nextitnet', 'sasrec'}

    if model_name == 'caser':
        model = Caser(
            hidden_size=args.hidden_factor,
            item_num=item_num,
            state_size=state_size,
            num_filters=args.num_filters,
            filter_sizes=args.filter_sizes,
            dropout_rate=args.dropout_rate
        )
    elif model_name == 'gru':
        model = GRU(
            hidden_size=args.hidden_factor,
            item_num=item_num,
            state_size=state_size,
        )
    elif model_name == 'nextitnet':
        model = NextItNet(
            hidden_size=args.hidden_factor,
            item_num=item_num,
            state_size=state_size,
            dilations=args.dilations,
            device=device
        )
    else:
        model = SASRec(
            hidden_size=args.hidden_factor,
            item_num=item_num,
            state_size=state_size,
            dropout=args.dropout_rate,
            device=device
        )
    model.state_dict = torch.load(models_path + '{}_model.pt'.format(model_name))
    test_acc = evaluate(model, args, 'test', state_size, item_num, device)
