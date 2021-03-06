import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

from src.decoder import Decoder
from src.encoder import Encoder

from src.perplexity import Perplexity

from torch.nn.utils.rnn import pack_padded_sequence


class RVAE_dilated(nn.Module):
    def __init__(self, params, embedding_matrix):
        super(RVAE_dilated, self).__init__()

        self.params = params

        self.word_embeddings = nn.Embedding(params.word_vocab_size, params.word_embed_size)
        self.word_embeddings.weight = Parameter(t.from_numpy(embedding_matrix).float(),
                                                requires_grad=False)
        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)

    def forward(self, drop_prob,
                encoder_input_tuple=None,
                use_cuda=False,
                z=None,
                inference=False):
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
        encoder_input = Variable(encoder_input, volatile=inference).cuda() if use_cuda else Variable(encoder_input, volatile=inference)
        encoder_input_matrix = self.word_embeddings(encoder_input)

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            packed_seq = pack_padded_sequence(encoder_input_matrix, lengths, batch_first=True)
            context = self.encoder.forward(packed_seq, batch_size)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu
            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()
        else:
            kld = None

        decoder_input = encoder_input_matrix
        out = self.decoder.forward(decoder_input, z, drop_prob)

        return encoder_input, out, kld

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, use_cuda, dropout):

        def train_one_batch(data_tuple):
            target, logits, kld = self.forward(drop_prob=dropout,
                                       encoder_input_tuple=data_tuple,
                                       use_cuda=use_cuda,
                                       z=None)

            batch_size = target.data.size()[0]
            sequence_length = target.data.size()[1]

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)
            loss = sequence_length * cross_entropy + kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return kld, loss

        return train_one_batch


    def validater(self, use_cuda, dropout):
        def validate(data_tuple):
            target, logits, kld = self.forward(drop_prob=dropout,
                                       encoder_input_tuple=data_tuple,
                                       use_cuda=use_cuda,
                                       z=None)
            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)
            return np.exp2(cross_entropy.data.cpu().numpy()[0])

        return validate




