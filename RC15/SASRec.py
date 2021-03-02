import torch.nn as nn
import pickle
import time
import sys
from utils import *
from evaluate import evaluate
from SASRecModules import *


class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

    def forward(self, states, len_states):
        inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
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
    print('Using {} for training'.format(device))

    if args.results_to_file:
        file_name = 'sasrec.txt'
        print('Outputs are saved to file {}'.format(args.results_path + file_name))
        sys.stdout = open(args.results_path + file_name, 'w')

    state_size, item_num = get_stats(args.data_path)
    train_loader = prepare_dataloader(args.data_path, args.batch_size)
    sasRec = SASRec(
        hidden_size=args.hidden_factor,
        item_num=item_num,
        state_size=state_size,
        dropout=args.dropout_rate,
        device=device
    )

    criterion = nn.CrossEntropyLoss()
    params1 = list(sasRec.parameters())
    optimizer = torch.optim.Adam(params1, lr=args.lr)

    sasRec.to(device)

    # Start training loop
    epoch_times = []
    total_step = 0
    best_val_acc = 0
    for epoch in range(0, args.epochs):
        sasRec.train()
        start_time = time.time()
        avg_loss = 0.
        counter = 0
        for state, len_state, action, next_state, len_next_state, is_buy, is_done in train_loader:
            counter += 1
            sasRec.zero_grad()
            supervised_out = sasRec(state.to(device).long(), len_state.to(device).long())
            supervised_loss = criterion(supervised_out, action.to(device).long())
            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()

            total_step += 1
            if total_step == 200:
                print('For 200 steps, {} seconds elapsed.'.format(time.time() - start_time))
                sys.stdout.flush()
                
            if total_step % 200 == 0:
                print('Model is Vanilla SASRec', )
                print('Supervised loss is %.3f' % (supervised_loss.item()))
                print('Epoch {}......Step: {}....... Loss: {}'.format(epoch, total_step, supervised_loss.item()))
                sys.stdout.flush()

            if total_step % 10000 == 0:
                print('Evaluating Vanilla SASREC Model')
                val_acc = evaluate(sasRec, args, 'val', state_size, item_num, device)
                sys.stdout.flush()
                sasRec.train()
                print('Current accuracy: ', val_acc)
                print('Best accuracy so far: ', best_val_acc)
                if val_acc > best_val_acc:
                    print('Main model is the best, so far!')
                    print('New best accuracy is: %.3f' % best_val_acc)
                    pickle.dump(sasRec, open(args.models_path + '/vanillaSASREC_model.pth', 'wb'))
        current_time = time.time()
        print('Epoch {}/{} Done'.format(epoch, args.epochs))
        print('Total Time Elapsed: {} seconds'.format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print('Total Training Time: {} seconds'.format(str(sum(epoch_times))))
    sys.stdout.close()
