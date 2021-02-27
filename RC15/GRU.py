import torch.nn as nn
import time
import pickle
import sys
import getpass
from utils import *
from evaluate import evaluate


class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, gru_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

    def forward_eval(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output


class Args:
    epochs = 30
    resume = 1
    batch_size = 256
    hidden_factor = 64
    lr = 0.01
    dropout_rate = 0.1
    data_path = '../data/'
    models_path = '../models/'
    results_path = '../results/'
    results_to_file = True


if __name__ == '__main__':
    # Network parameters
    args = Args()

    device = set_device()
    print('Using {} For Training'.format(torch.cuda.get_device_name()))

    if args.results_to_file:
        file_name = 'gru_sep_eval.txt'
        print('Outputs are saved to file {}'.format(args.results_path + file_name))
        sys.stdout = open(args.results_path + file_name, 'w')

    state_size, item_num = get_stats(args.data_path)
    train_loader = prepare_dataloader(args.data_path, args.batch_size)
    GRU1 = GRU(hidden_size=args.hidden_factor,
               item_num=item_num,
               state_size=state_size,
               )

    criterion = nn.CrossEntropyLoss()
    params = list(GRU1.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    GRU1.to(device)

    # Start training loop
    epoch_times = []
    total_step = 0
    best_val_acc = 7.5
    for epoch in range(0, args.epochs):
        start_time = time.time()
        avg_loss = 0.
        counter = 0
        for state, len_state, action, next_state, len_next_state, is_buy, is_done in train_loader:
            counter += 1

            GRU1.zero_grad()

            supervised_out = GRU1(state.to(device).long(), len_state.to(device).long())
            supervised_loss = criterion(supervised_out, action.to(device).long())

            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()

            total_step += 1

            if total_step == 200:
                print('For 200 steps, {} seconds elapsed.'.format(time.time() - start_time))
                sys.stdout.flush()

            if total_step % 200 == 0:
                print('Model is Vanilla GRU')
                print('Supervised loss is %.3f' % (supervised_loss.item()))
                print('Epoch {}......Step: {}....... Loss: {}'.format(epoch, total_step, supervised_loss.item()))
                sys.stdout.flush()

            if total_step % 10000 == 0:
                print('Evaluating Vanilla GRU Model')
                val_acc = evaluate(GRU1, args, 'val', state_size, item_num, device)
                GRU1.train()
                print('Current accuracy: ', val_acc)
                print('Best accuracy so far: ', best_val_acc)
                if val_acc > best_val_acc:
                    print('Main model is the best, so far!')
                    print('New best accuracy is: %.3f' % best_val_acc)
                    pickle.dump(GRU1, open(args.models_path + '/GRU_model.pth', 'wb'))
        current_time = time.time()
        print('Epoch {}/{} Done'.format(epoch, args.epochs))
        print('Total Time Elapsed: {} seconds'.format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print('Total Training Time: {} seconds'.format(str(sum(epoch_times))))
    sys.stdout.close()
