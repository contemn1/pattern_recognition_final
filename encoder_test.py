from padding_data_loader import create_data_loader
from parameters import Parameters
from encoder import Encoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from rvae_dilated import RVAE_dilated
from torch.optim import Adam
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RVAE_dilated')
    parser.add_argument('--num-iterations', type=int, default=25000, metavar='NI',
                        help='num iterations (default: 25000)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 45)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
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

    args = parser.parse_args()

    print(args)

    data_loader, num_words = create_data_loader(text_path="data/nips_test_sorted.txt",
                                     glove_path="/Users/zxj/Downloads/nips_data/glove_nips.txt",
                                     batch_size=args.batch_size)
    parameters = Parameters(num_of_words=num_words)
    rvae = RVAE_dilated(params=parameters)
    adam_optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    current_trainer = rvae.trainer(optimizer=adam_optimizer, data_loader=data_loader)

    for iteration in range(2):
        current_trainer(args.use_cuda, args.dropout)