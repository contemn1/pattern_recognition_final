import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.params = params

        self.kernels = [Parameter(t.Tensor(out_chan, in_chan, width).normal_(0, 0.05))
                        for out_chan, in_chan, width in params.decoder_kernels]
        self._add_to_parameters(self.kernels, 'decoder_kernel')

        self.biases = [Parameter(t.Tensor(out_chan).normal_(0, 0.05))
                       for out_chan, in_chan, width in params.decoder_kernels]
        self._add_to_parameters(self.biases, 'decoder_bias')

        self.out_size = self.params.decoder_kernels[-1][0]

        self.fc = nn.Linear(self.out_size, self.params.word_vocab_size)

    def forward(self, decoder_input):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]

        :return: unnormalized logits of sentense words distribution probabilities
                 with shape of [batch_size, seq_len, word_vocab_size]
        """

        # x is tensor with shape [batch_size, input_size=in_channels, seq_len=input_width]
        seq_len = decoder_input.size()[1]
        x = decoder_input.transpose(1, 2).contiguous()

        for layer, kernel in enumerate(self.kernels):
            # apply conv layer with non-linearity and drop last elements of sequence to perfrom input shifting
            x = F.conv1d(x, kernel,
                         bias=self.biases[layer],
                         dilation=self.params.decoder_dilations[layer],
                         padding=self.params.decoder_paddings[layer])

            x_width = x.size()[2]
            x = x[:, :, :(x_width - self.params.decoder_paddings[layer])].contiguous()

            x = F.relu(x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, self.out_size)
        x = self.fc(x)
        result = x.view(-1, seq_len, self.params.word_vocab_size)

        return result

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)

