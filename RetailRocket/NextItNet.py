import time
import torch.nn as nn
import sys
from utils import *
from evaluate import evaluate
from NextItNetModules import ResidualBlock


class NextItNet(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dilations, device):
        super(NextItNet, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.dilations = eval(dilations)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # Initialize embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Convolutional Layers
        self.cnns = nn.ModuleList([
            ResidualBlock(
                in_channels=1,
                residual_channels=hidden_size,
                kernel_size=3,
                dilation=i,
                hidden_size=hidden_size) for i in self.dilations
        ])
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(device)
        emb *= mask
        conv_out = emb
        for cnn in self.cnns:
            conv_out = cnn(conv_out)
            conv_out *= mask
        state_hidden = extract_axis_1(conv_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(device)
        emb *= mask
        conv_out = emb
        for cnn in self.cnns:
            conv_out = cnn(conv_out)
            conv_out *= mask
        state_hidden = extract_axis_1(conv_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output


class Args:
    r_click = 0.2
    r_buy = 1.0
    epochs = 30
    resume = 1
    batch_size = 256
    lr = 0.01
    dropout_rate = 0.1
    hidden_factor = 64
    num_filters = 16
    dilations = '[1, 2, 1, 2, 1, 2]'
    data_path = '../data/'
    models_path = '../models/'
    results_path = '../results/'
    results_to_file = True


if __name__ == '__main__':
    # Network parameters
    args = Args()

    device = set_device()
    print('Using {} for training'.format(device))

    if args.results_to_file:
        file_name = 'nextitnet.txt'
        print('Outputs are saved to file {}'.format(args.results_path + file_name))
        sys.stdout = open(args.results_path + file_name, 'w')

    state_size, item_num = get_stats(args.data_path)
    train_loader = prepare_dataloader(args.data_path, args.batch_size)

    nextItNet = NextItNet(
        hidden_size=args.hidden_factor,
        item_num=item_num,
        state_size=state_size,
        dilations=args.dilations,
        device=device
    )

    criterion = nn.CrossEntropyLoss()
    params1 = list(nextItNet.parameters())
    optimizer = torch.optim.Adam(params1, lr=args.lr)

    nextItNet.to(device)

    # Start training loop
    epoch_times = []
    total_step = 0
    best_val_acc = 0
    for epoch in range(0, args.epochs):
        nextItNet.train()
        start_time = time.time()
        avg_loss = 0.
        counter = 0
        for state, len_state, action, next_state, len_next_state, is_buy, is_done in train_loader:
            counter += 1

            nextItNet.zero_grad()
            supervised_out = nextItNet(state.to(device).long(), len_state.to(device).long())
            supervised_loss = criterion(supervised_out, action.to(device).long())
            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()

            total_step += 1
            if total_step == 200:
                print('For 200 steps, {} seconds elapsed.'.format(time.time() - start_time))
                sys.stdout.flush()

            if total_step % 200 == 0:
                print('Model is NextItNet',)
                print('Supervised loss is %.3f' % (supervised_loss.item()))
                print('Epoch {}......Step: {}....... Loss: {}'.format(epoch, total_step,
                                                                      supervised_loss.item()))
            if total_step % 10000 == 0:
                print('Evaluating Vanilla NextItNet Model')
                val_acc = evaluate(nextItNet, args, 'val', state_size, item_num, device)
                nextItNet.train()
                print('Current accuracy: ', val_acc)
                print('Best accuracy so far: ', best_val_acc)
                if val_acc > best_val_acc:
                    print('Main model is the best, so far!')
                    print('New best accuracy is: %.3f' % best_val_acc)
                    torch.save(nextItNet.state_dict(), args.models_path + '/nextitnet_model.pt')

        current_time = time.time()
        print('Epoch {}/{} Done'.format(epoch, args.epochs))
        print('Total Time Elapsed: {} seconds'.format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print('Total Training Time: {} seconds'.format(str(sum(epoch_times))))
    sys.stdout.close()
