import torch.nn as nn
import time
import sys
from utils import *
from evaluate import evaluate


class Caser(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
                 dropout_rate):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

    def forward_eval(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)

        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output


class Args:
    r_click = 0.2
    r_buy = 1.0
    epochs = 30
    resume = 1
    batch_size = 256
    hidden_factor = 64
    lr = 0.01
    dropout_rate = 0.1
    num_filters = 16
    filter_sizes = '[2,3,4]'
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
        file_name = 'caser.txt'
        print('Outputs are saved to file {}'.format(args.results_path + file_name))
        sys.stdout = open(args.results_path + file_name, 'w')

    state_size, item_num = get_stats(args.data_path)
    train_loader = prepare_dataloader(args.data_path, args.batch_size)

    caser = Caser(
        hidden_size=args.hidden_factor,
        item_num=item_num,
        state_size=state_size,
        num_filters=args.num_filters,
        filter_sizes=args.filter_sizes,
        dropout_rate=args.dropout_rate
    )
    criterion = nn.CrossEntropyLoss()
    params = list(caser.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    caser.to(device)

    # Start training loop
    epoch_times = []
    total_step = 0
    best_val_acc = 0
    for epoch in range(0, args.epochs):
        caser.train()
        start_time = time.time()
        avg_loss = 0.
        counter = 0
        for state, len_state, action, next_state, len_next_state, is_buy, is_done in train_loader:
            counter += 1

            caser.zero_grad()
            supervised_out = caser(state.to(device).long(), len_state.to(device).long())
            supervised_loss = criterion(supervised_out, action.to(device).long())
            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()

            total_step += 1
            if total_step == 200:
                print('For 200 steps, {} seconds elapsed.'.format(time.time() - start_time))
                sys.stdout.flush()

            if total_step % 200 == 0:
                print('Model is Vanilla Caser')
                print('Supervised loss is %.3f' % (supervised_loss.item()))
                print('Epoch {}......Step: {}....... Loss: {}'.format(epoch, total_step, supervised_loss.item()))
                sys.stdout.flush()

            if total_step % 10000 == 0:
                print('Evaluating Vanilla Caser Model')
                val_acc = evaluate(caser, args, 'val', state_size, item_num, device)
                caser.train()
                print('Current accuracy: ', val_acc)
                print("Best accuracy so far: ", best_val_acc)
                if val_acc > best_val_acc:
                    print('Main model is the best, so far!')
                    print('New best accuracy is: %.3f' % best_val_acc)
                    torch.save(caser.state_dict(), args.models_path + '/caser_model.pt')

        current_time = time.time()
        print('Epoch {}/{} Done'.format(epoch, args.epochs))
        print('Total Time Elapsed: {} seconds'.format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print('Total Training Time: {} seconds'.format(str(sum(epoch_times))))
    sys.stdout.close()
