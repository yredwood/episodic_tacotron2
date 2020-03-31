from math import sqrt
import numpy as np
from numpy import finfo
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from modules import GST
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
        
        self.decoder_rnn = nn.LSTMCell(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
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

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs)
        if len(gate_outputs.size()) > 1:
            gate_outputs = gate_outputs.transpose(0, 1)
        else:
            gate_outputs = gate_outputs[None]
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments


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

            h, c = self.decoder_rnn(decoder_input, (h, c))
            c = F.dropout(c, self.p_decoder_dropout, self.training)

            output_context = torch.cat((h, context), dim=1)

            frame_prediction = self.linear_projection(output_context)
            gate_prediction = self.gate_layer(output_context)
            
            mel_outputs.append(frame_prediction)
            gate_outputs.append(gate_prediction)


            if torch.sigmoid(gate_output.data) > self.gate_trheshold and target is None:
                # inference break condition
                break
        
        # parse output
        gate_outputs = torch.stack(gate_outputs).transpose(0,1)
        mel_outputs = torch.stack(mel_outputs).transpose(0,1)
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels).transpose(1,2)
        return mel_outputs, gate_outputs, None


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
        #f0_padded = to_gpu(f0_padded).float()
        f0_padded = None
        return ((text_padded, input_lengths, mel_padded, max_len,
                 output_lengths, speaker_ids, f0_padded),
                (mel_padded, gate_padded))

        return mel_padded, (mel_padded, gate_padded)

    def forward(self, inputs):
        mel_padded = inputs
        target_encoded = self.gst(mel_padded)
        
        recon, gates = self.decoder(target_encoded, mel_padded)
        postnet_out = self.postnet(recon)
        postnet_out = recon + postnet_out
        # masked_output
        pdb.set_trace()
        return 







class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        if hparams.with_gst:
            self.gst = GST(hparams)

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
        #f0_padded = to_gpu(f0_padded).float()
        f0_padded = None
        return ((text_padded, input_lengths, mel_padded, max_len,
                 output_lengths, speaker_ids, f0_padded),
                (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, \
            output_lengths, speaker_ids, f0s = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        embedded_text = self.encoder(embedded_inputs, input_lengths)
        #embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
        embedded_gst = self.gst(targets)
        embedded_gst = embedded_gst.repeat(1, embedded_text.size(1), 1)
        #embedded_speakers = embedded_speakers.repeat(1, embedded_text.size(1), 1)
#        embedded_speakers = embedded_gst.new_zeros(embedded_gst.size(0),
#                embedded_text.size(1), self.speaker_embedding_dim)
        
        embedded_text = embedded_text.new_zeros(embedded_text.size())
        encoder_outputs = torch.cat(
            (embedded_text, embedded_gst), dim=2) #, embedded_speakers), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths, f0s=f0s)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        text, style_input, speaker_ids, f0s = inputs
        embedded_inputs = self.embedding(text).transpose(1, 2)
        embedded_text = self.encoder.inference(embedded_inputs)
        embedded_text = embedded_text.new_zeros(embedded_text.size())
        #embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
        if hasattr(self, 'gst'):
            if isinstance(style_input, int):
                query = torch.zeros(1, 1, self.gst.encoder.ref_enc_gru_size).cuda()
                GST = torch.tanh(self.gst.stl.embed)
                key = GST[style_input].unsqueeze(0).expand(1, -1, -1)
                embedded_gst = self.gst.stl.attention(query, key)
            else:
                embedded_gst = self.gst(style_input)

        #embedded_speakers = embedded_speakers.repeat(1, embedded_text.size(1), 1)
#        embedded_speakers = embedded_gst.new_zeros(embedded_gst.size(0),
#                embedded_text.size(1), self.speaker_embedding_dim)

        if hasattr(self, 'gst'):
            embedded_gst = embedded_gst.repeat(1, embedded_text.size(1), 1)
            encoder_outputs = torch.cat(
                (embedded_text, embedded_gst), dim=2) # , embedded_speakers), dim=2)
        else:
#            encoder_outputs = torch.cat(
#                (embedded_text, embedded_speakers), dim=2)
            encoder_outputs = embedded_text

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, f0s)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

    def inference_noattention(self, inputs):
        text, style_input, speaker_ids, f0s, attention_map = inputs
        embedded_inputs = self.embedding(text).transpose(1, 2)
        embedded_text = self.encoder.inference(embedded_inputs)
        #embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
        if hasattr(self, 'gst'):
            if isinstance(style_input, int):
                query = torch.zeros(1, 1, self.gst.encoder.ref_enc_gru_size).cuda()
                GST = torch.tanh(self.gst.stl.embed)
                key = GST[style_input].unsqueeze(0).expand(1, -1, -1)
                embedded_gst = self.gst.stl.attention(query, key)
            else:
                embedded_gst = self.gst(style_input)

        #embedded_speakers = embedded_speakers.repeat(1, embedded_text.size(1), 1)
#        embedded_speakers = embedded_gst.new_zeros(embedded_gst.size(0),
#                embedded_text.size(1), self.speaker_embedding_dim)
        if hasattr(self, 'gst'):
            embedded_gst = embedded_gst.repeat(1, embedded_text.size(1), 1)
            encoder_outputs = torch.cat(
                (embedded_text, embedded_gst), dim=2) #, embedded_speakers), dim=2)
        else:
#            encoder_outputs = torch.cat(
#                (embedded_text, embedded_speakers), dim=2)
            encoder_outputs = embedded_text

        mel_outputs, gate_outputs, alignments = self.decoder.inference_noattention(
            encoder_outputs, f0s, attention_map)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
