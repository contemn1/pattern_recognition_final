import argparse

import torch
from src.padding_data_loader import create_new_data_loader
from src.parameters import Parameters
from src.rvae_dilated import RVAE_dilated
from torch.optim import Adam

result_path = "results/"


def save_checkpoint(state, filename='checkpoint_ecpch{0}.tar'):
    filename = result_path + filename.format(state['epoch'])
    torch.save(state, filename)


def train_model(args, model_path=None):
    loader, num_words, embedding_matrix = create_new_data_loader(args)
    parameters = Parameters(num_of_words=num_words, use_cuda=args.use_cuda)
    rvae = RVAE_dilated(params=parameters, embedding_matrix=embedding_matrix)
    adam_optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)
    if args.use_cuda:
        rvae = rvae.cuda()

    if model_path:
        check_point = torch.load(model_path)
        rvae.load_state_dict(check_point['state_dict'])
        adam_optimizer.load_state_dict(check_point['optimizer'])

    current_trainer = rvae.trainer(optimizer=adam_optimizer, data_loader=loader)
    for iteration in range(args.num_iterations):
        kld_list, loss_list = current_trainer(args.use_cuda, args.dropout)
        if iteration % 3 == 0:
            save_checkpoint({
            'epoch': iteration + 1,
            'state_dict': rvae.state_dict(),
            'optimizer': adam_optimizer.state_dict(),
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RVAE_dilated')
    parser.add_argument('--num-iterations', type=int, default=25000, metavar='NI',
                        help='num iterations (default: 25000)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ppl-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')
    parser.add_argument('--embedding-dimension', type=int, default=300, metavar='ED',
                        help='embedding dimension (default: 300)')
    parser.add_argument('--train-path', type=str,
                        default='data/nips_train_sorted.txt',
                        help='default train path is data/nips_train_sorted')
    parser.add_argument('--glove-path', type=str,
                        default='data/glove_nips.txt',
                        help='default glove path is data/glove_nips.txt')

    args = parser.parse_args()

    train_model(args)