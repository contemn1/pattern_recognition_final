import numpy as np
import torch
import torch.nn.functional as F
from src.avb_model.avb_decoder import Decoder
from torch import autograd
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence

from src.avb_model.avb_encoder import Encoder


class AVB(nn.Module):

    def __init__(self, params, embedding_matrix, noise_size, batch_size):
        super(AVB, self).__init__()

        self.params = params

        self.word_embeddings = nn.Embedding(params.word_vocab_size, params.word_embed_size)
        self.word_embeddings.weight = Parameter(torch.from_numpy(embedding_matrix).float(),
                                                requires_grad=False)
        self.encoder = Encoder(self.params, noise_size=noise_size)

        self.context_to_latent = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        zeros = torch.zeros((batch_size, self.params.latent_variable_size))
        ones = torch.ones((batch_size, self.params.latent_variable_size))
        if params.use_cuda:
            zeros = zeros.cuda()
            ones = ones.cuda()

        self.z_mean = Variable(zeros, requires_grad=True)
        self.z_var = Variable(ones, requires_grad=True)
        self.z_std = Variable(ones, requires_grad=True)

        self.decoder = Decoder(self.params)

        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        if params.use_cuda:
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()

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
        encoder_input_matrix = self.word_embeddings(encoder_input)

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            packed_seq = pack_padded_sequence(encoder_input_matrix, lengths, batch_first=True)
            context = self.encoder.forward(packed_seq, batch_size)
            z = self.context_to_latent(context)

        fake_z = torch.randn((batch_size, self.params.latent_variable_size))
        if use_cuda:
            fake_z = fake_z.cuda()

        fake_z = Variable(fake_z, requires_grad=True)

        fake = create_decoder_input(encoder_input_matrix, fake_z, drop_prob, self.params.latent_variable_size)

        real = create_decoder_input(encoder_input_matrix, z, drop_prob, self.params.latent_variable_size)

        out = self.decoder.forward(real)

        return encoder_input, out, real, fake.detach()

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer_vae, discriminator, optimizer_discriminator, data_loader):
        def train(use_cuda, dropout):
            loss_list = []
            kld_list = []
            for data_tuple in data_loader:
                target, logits, real, fake = self.forward(drop_prob=dropout,
                                       encoder_input_tuple=data_tuple,
                                       use_cuda=use_cuda,
                                       z=None)
                batch_size = target.data.size()[0]
                sequence_length = target.data.size()[1]


                logits = logits.view(-1, self.params.word_vocab_size)
                target = target.view(-1)
                cross_entropy = F.cross_entropy(logits, target)
                kld = discriminator.forward(real).mean()
                loss = sequence_length*cross_entropy + kld
                kld_number =kld.data.cpu().numpy()[0]
                loss_number = loss.data.cpu().numpy()[0]
                kld_list.append(kld_number)
                loss_list.append(loss_number)
                print(kld_number, loss_number, np.exp2(cross_entropy.data.cpu()[0]))
                optimizer_vae.zero_grad()
                loss.backward()
                optimizer_vae.step()

                discriminator.zero_grad()
                detached_real = real.detach()
                d_real = discriminator(detached_real)
                d_real = d_real.mean()
                d_real.backward(self.mone)
                d_fake = discriminator(fake)
                d_fake = d_fake.mean()
                d_fake.backward(self.one)
                gradient_penalty = calc_gradient_penalty(discriminator, detached_real.data, fake.data, batch_size, use_cuda)
                gradient_penalty.backward()
                D_cost = d_fake - d_real + gradient_penalty
                Wasserstein_D = d_real - d_fake
                optimizer_discriminator.step()

            return kld_list, loss_list

        return train


LAMBDA = 10


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, use_cuda):
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def create_decoder_input(decoder_input, z, drop_prob, latent_variable_size):
    [batch_size, seq_len, _] = decoder_input.size()
    z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, latent_variable_size)
    decoder_input = torch.cat([decoder_input, z], 2)
    decoder_input = F.dropout(decoder_input, drop_prob)
    return decoder_input
