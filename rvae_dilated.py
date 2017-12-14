import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from decoder import Decoder
from encoder import Encoder

from perplexity import Perplexity

from functional import kld_coef, parameters_allocation_check, fold
from torch.nn.utils.rnn import pack_padded_sequence


class RVAE_dilated(nn.Module):
    def __init__(self, params):
        super(RVAE_dilated, self).__init__()

        self.params = params

        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)

    def forward(self, drop_prob,
                encoder_input_tuple=None,
                use_cuda=False,
                z=None):
        """
        :param encoder_input_tuple: An tensor with shape of [batch_size, seq_len] of Long type

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param use_cuda: whether to use gpu


        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 kld loss estimation
        """
        encoder_input, lengths = encoder_input_tuple
        batch_size = encoder_input.size()[0]
        encoder_input = Variable(encoder_input).cuda() if use_cuda else Variable(encoder_input)

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            packed_seq = pack_padded_sequence(encoder_input, lengths, batch_first=True)

            context = self.encoder.forward(packed_seq, batch_size)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu
            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean() / self.params.latent_variable_size
            print(kld)
        else:
            kld = None

        decoder_input = encoder_input
        out = self.decoder.forward(decoder_input, z, drop_prob)

        return encoder_input, out, kld

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, data_loader):
        def train(use_cuda, dropout):
            losses = []
            loss = 0
            kld = 0
            for data_tuple in data_loader:
                optimizer.zero_grad()
                target, logits, kld = self.forward(drop_prob=dropout,
                                       encoder_input_tuple=data_tuple,
                                       use_cuda=use_cuda,
                                       z=None)

                cross_entropy = F.binary_cross_entropy_with_logits(logits, target)
                loss = cross_entropy + kld
                loss.backward()
                optimizer.step()

            return kld, loss

        return train

