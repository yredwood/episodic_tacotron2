from math import sqrt
import numpy as np
from numpy import finfo
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from modules import GST_las as GST
import pdb

drop_rate = 0.5

def load_model(hparams):
    #model = Tacotron2(hparams).cuda()
    model = TacotronAutoencoder(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min
    return model

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=drop_rate, training=True)
        return x

class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), drop_rate, self.training)
        x = F.dropout(self.convolutions[-1](x), drop_rate, self.training)

        return x


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, attention_weights=None):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        if attention_weights is None:
            alignment = self.get_alignment_energies(
                attention_hidden_state, processed_memory, attention_weights_cat)

            if mask is not None:
                alignment.data.masked_fill_(mask, self.score_mask_value)

            attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.token_embedding_size 
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.p_teacher_forcing = hparams.p_teacher_forcing



        # reference mel layers

        self.prenet = Prenet(hparams.n_mel_channels * hparams.n_frames_per_step,
                [hparams.prenet_dim, hparams.prenet_dim])


        self.attention_rnn = nn.LSTMCell(
                hparams.prenet_dim + self.encoder_embedding_dim,
                hparams.attention_rnn_dim)

        self.attention_layer = Attention(
                hparams.attention_rnn_dim, self.encoder_embedding_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size)
        
        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask


    def forward(self, context, target=None):
        frame0 = self.get_go_frame(context)
        h = Variable(context.data.new(context.size(0), self.decoder_rnn_dim).zero_())
        c = Variable(context.data.new(context.size(0), self.decoder_rnn_dim).zero_())

        mel_outputs, gate_outputs, alignments = [], [], []
        if target is not None:
            # training mode
            target = target.transpose(1, 2) # B,80,t -> B,t,80
            target = target.view(target.size(0),
                    int(target.size(1) / self.n_frames_per_step), -1)
            target = target.transpose(0,1) # t,b,80
            self.max_decoder_steps = target.size(0)

        for t in range(self.max_decoder_steps):
            if len(mel_outputs) == 0:
                decoder_input = frame0
            elif np.random.uniform(0.0, 1.0) <= self.p_teacher_forcing and target is not None:
                decoder_input = target[len(mel_outputs)]
            else:
                decoder_input = mel_outputs[-1]
            
            decoder_input = torch.cat((self.prenet(decoder_input), context), -1)
            h, c = self.decoder_rnn(decoder_input, (h, c))
            c = F.dropout(c, self.p_decoder_dropout, self.training)

            output_context = torch.cat((h, context), dim=1)

            frame_prediction = self.linear_projection(output_context)
            gate_prediction = self.gate_layer(output_context)
            
            mel_outputs.append(frame_prediction)
            gate_outputs.append(gate_prediction)

            if (target is None) and (torch.sigmoid(gate_prediction) > self.gate_threshold).all():
                # inference break condition
                break
        
        # parse output
        gate_outputs = torch.stack(gate_outputs).transpose(0,1).contiguous()
        mel_outputs = torch.stack(mel_outputs).transpose(0,1)
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels).contiguous() # bsz,T,d
        #mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels).transpose(1,2)
        return mel_outputs, gate_outputs


class TacotronAutoencoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels =hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.gst = GST(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, speaker_ids, f0_padded = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        speaker_ids = to_gpu(speaker_ids.data).long()

        return (mel_padded, output_lengths), (mel_padded, gate_padded)


    def masked_output(self, x, length, value=0.0):
        mask = ~get_mask_from_lengths(length)
        # each input should be B,d,t
        x.data.masked_fill_(mask.unsqueeze(2), value)
        return x

    def forward(self, inputs):
        mel_padded, mel_length = inputs
        target_encoded, target_lengths = self.gst(mel_padded).squeeze(1)
        
        mel_out, gate_out = self.decoder(target_encoded, mel_padded)
        postnet_out = self.postnet(mel_out.transpose(-1,-2))
        postnet_out = mel_out + postnet_out.transpose(-1,-2)

        # masked_output
        mel_out = self.masked_output(mel_out, mel_length, 0.0)
        postnet_out = self.masked_output(postnet_out, mel_length, 0.0)
        #gate_out = self.masked_output(gates, mel_length, 1e3)
        zero_alignments = mel_out.new_zeros(10,10,10)

        return mel_out.transpose(-1,-2), postnet_out.transpose(-1,-2), gate_out, zero_alignments

    def inference(self, inputs):
        mel_padded = inputs
        target_encoded = self.gst(mel_padded).squeeze(1)
        
        mel_out, gate_out = self.decoder(target_encoded)
        postnet_out = self.postnet(mel_out.transpose(-1,-2))
        postnet_out = mel_out + postnet_out.transpose(-1,-2)
        return mel_out.transpose(-1,-2), postnet_out.transpose(-1,-2), gate_out
