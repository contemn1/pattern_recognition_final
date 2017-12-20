from torch import nn


class AVB(nn.Module):

    def __init__(self, params, embedding_matrix):
        super(AVB, self).__init__()

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

    def trainer(self, optimizer, data_loader):
        def train(use_cuda, dropout):
            loss = 0
            kld = 0
            for data_tuple in data_loader:
                target, logits, kld = self.forward(drop_prob=dropout,
                                       encoder_input_tuple=data_tuple,
                                       use_cuda=use_cuda,
                                       z=None)

                logits = logits.view(-1, self.params.word_vocab_size)
                target = target.view(-1)
                cross_entropy = F.cross_entropy(logits, target)

                loss = 79 * cross_entropy + kld
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            return kld, loss

        return train
