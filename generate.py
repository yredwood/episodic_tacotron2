import sys
sys.path.append('waveglow/')

import os
from shutil import copyfile
import numpy as np
import scipy as sp
from scipy.io.wavfile import write
import pandas as pd
import librosa
import torch
import matplotlib
import matplotlib.pyplot as plt

from audio_processing import griffin_lim
from hparams import create_hparams
from model import EpisodicTacotron, Tacotron2, load_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from data_utils import EpisodicLoader, EpisodicCollater, EpisodicBatchSampler
from text import cmudict, text_to_sequence, sequence_to_text

import pdb


# ========== parameters ===========
checkpoint_path = 'models/tst_tacotron2_161616_single_2_pretrained/checkpoint_18000'
waveglow_path = 'models/waveglow_256channels_v4.pt'
#waveglow_path = '/home/mike/models/waveglow/waveglow_80000'
audio_path = 'filelists/libri100_val.txt'
num_support_save = 2

test_text_list = [
    'AITRICS leads the race to optimized precision care, strengthening and trust.',
    'Our mission is to improve patient outcomes.',
    '"Oh, I believe everything I am told," said the Caterpillar.',
    'I think you must be deceived so far.',
    'Did you cross the bridge at that time?',
    'She did not turn her head towards him, although, having such a long and slender neck,' \
            + 'she could have done so with very little trouble',
]

#supportset_sid = '2952'  # m
supportset_sid = '1069' # f 
output_root = 'audios'

output_dir = os.path.join(
        output_root,
        '-'.join(checkpoint_path.split('/')[-2:]) + '-' + supportset_sid,
)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_mel(path):
    audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
    audio = torch.from_numpy(audio)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec

def load_dataloader(hparams, audio_path):
    if not hparams.episodic_training:
        dataloader = TextMelLoader(audio_path, hparams)
        datacollate = TextMelCollate(1)
    else:
        dataloader = EpisodicLoader(audio_path, hparams)
        datacollate = EpisodicCollater(1, hparams)
    
    return dataloader, datacollate

def save_figure(mel_pred, attention, fname, description='None'):
    gridsize = (3,1)
    fig = plt.figure(figsize=(20,20))
    ax1 = plt.subplot2grid(gridsize, (0,0))
    ax2 = plt.subplot2grid(gridsize, (1,0), rowspan=2)

    ax1.imshow(mel_pred)
    ax2.imshow(attention)
    ax2.set_title(description)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)


hparams = create_hparams()
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)

# load tacotron2 model
model = load_model(hparams).cuda().eval()
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

# load waveglow model
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()

# dataloader
arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
dataloader, datacollate = load_dataloader(hparams, audio_path)

# only for episodic
batch_sampler = EpisodicBatchSampler(dataloader.sid_to_index, hparams, shuffle=False)
for batch_idx in batch_sampler:
    _batch = datacollate([dataloader[i] for i in batch_idx])
    _, _, sid = dataloader.audiopaths_and_text[batch_idx[0]]
    if sid == supportset_sid:
        break
for i in range(num_support_save):
    ref_idx = _batch['support']['idx'].data.tolist().index(i)
    batch, _ = model.parse_batch(_batch.copy())
    # 1. save reference wav and synthesized wav from the same text
    audiopath, test_text, speaker = dataloader.audiopaths_and_text[batch_idx[i]]
    #copyfile(audiopath, os.path.join(output_dir, 'ref_true.wav'))
    fname_wav = os.path.join(output_dir, 'ref_true_{}.wav'.format(i))
    mel_outputs_postnet = batch['support']['mel_padded'][ref_idx:ref_idx+1]
    # remove pad
    #mel_len = int(batch['support']['f0_padded'][ref_idx].sum().item())
    mel_len = (mel_outputs_postnet.mean(1) != 0).sum()
    mel_outputs_postnet = mel_outputs_postnet[:,:,:mel_len]
    audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:,0]
    write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
    save_figure(mel_outputs_postnet[0].data.cpu().numpy(),
            np.zeros((10,10)), fname_wav.replace('.wav', '.png'),
            description=test_text)
    text_encoded = torch.LongTensor(
            text_to_sequence(test_text,
                hparams.text_cleaners,
                arpabet_dict)
            )[None,:].cuda()
    text_lengths = torch.LongTensor(
            [len(text_encoded)]).cuda()

    input_dict = {'query': {'text_padded': text_encoded, 'input_lengths': text_lengths},
            'support': batch['support']}

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(input_dict)
        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:,0]

    fname_wav = os.path.join(output_dir, 'ref_pred_{}.wav'.format(i))
    write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
    save_figure(mel_outputs_postnet[0].data.cpu().numpy(), 
            alignments[0].data.cpu().numpy(), fname_wav.replace('.wav', '.png'), 
        description=test_text)
    print (test_text)


for tidx, test_text in enumerate(test_text_list):
    text_encoded = torch.LongTensor(
            text_to_sequence(test_text,
                hparams.text_cleaners,
                arpabet_dict)
            )[None,:].cuda()
    text_lengths = torch.LongTensor(
            [len(text_encoded)]).cuda()

    input_dict = {'query': {'text_padded': text_encoded, 'input_lengths': text_lengths},
            'support': batch['support']}

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(input_dict)
        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:,0]
    
    fname_wav = os.path.join(output_dir, '{}.wav'.format(tidx))
    write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
    save_figure(mel_outputs_postnet[0].data.cpu().numpy(), 
            alignments[0].data.cpu().numpy().T, fname_wav.replace('.wav', '.png'), 
        description=test_text)
    print (test_text)



















#
