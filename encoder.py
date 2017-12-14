import torch as t
import torch.nn as nn
import torch.nn.functional as F
from highway import  Highway


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        self.rnn = nn.LSTM(input_size=self.params.word_embed_size,
                           hidden_size=self.params.encoder_rnn_size,
                           num_layers=self.params.encoder_num_layers,
                           batch_first=True,
                           bidirectional=True)


    def forward(self, input_seq, batch_size):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        _, (final_state, _) = self.rnn(input_seq)

        final_state = final_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = t.cat([h_1, h_2], 1)
        return final_state
