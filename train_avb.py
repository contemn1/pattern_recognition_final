import argparse
import os
import torch
import sys
print(sys.path)

from src.parameters import Parameters
from torch.optim import Adam

from src.avb_model.avb import AVB
from src.avb_model.discriminator import Discriminator
from src.padding_data_loader import create_new_data_loader

result_path = "/home/zxj/Documents/pattern_recognition_final/results"


def save_checkpoint(state, filename='checkpoint_ecpch{0}.tar'):
    filename = result_path + filename.format(state['epoch'])
    torch.save(state, filename)


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
                        default='./data/nips_train_sorted.txt',
                        help='default train path is data/nips_train_sorted')
    parser.add_argument('--glove-path', type=str,
                        default='./data/glove_nips.txt',
                        help='default glove path is data/glove_nips.txt')
    parser.add_argument('--noise-size', type=int, default=50, metavar='NS',
                        help='noise size (default: 50)')



    args = parser.parse_args()

    loader, num_words, embedding_matrix = create_new_data_loader(args)
    parameters = Parameters(num_of_words=num_words, use_cuda=args.use_cuda)
    cur_avb = AVB(params=parameters, embedding_matrix=embedding_matrix,
                  noise_size=args.noise_size,
                  batch_size=args.batch_size)
    curr_discriminator = Discriminator(params=parameters)
    if args.use_cuda:
        cur_avb = cur_avb.cuda()
        curr_discriminator = curr_discriminator.cuda()

    optimizer_vae = Adam(cur_avb.learnable_parameters(), args.learning_rate)
    optimizer_discriminator = Adam(curr_discriminator.parameters(), args.learning_rate)

    current_trainer = cur_avb.trainer(optimizer_vae=optimizer_vae,
                                      discriminator=curr_discriminator,
                                      optimizer_discriminator=optimizer_discriminator,
                                      data_loader=loader)
    results = []
    for iteration in range(args.num_iterations):
        kld_list, loss_list = current_trainer(args.use_cuda, args.dropout)
        if iteration % 10 == 0:
            save_checkpoint({
            'epoch': iteration + 1,
            'vae_state_dict': cur_avb.state_dict(),
            'discriminator_state_dict': curr_discriminator.state_dict(),
            'optimizer_vae': optimizer_vae.state_dict(),
            'optimizer_discriminator':optimizer_discriminator.state_dict()
            })
