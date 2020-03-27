import sys
sys.path.append('waveglow/')

import os
from shutil import copyfile
import numpy as np
import scipy as sp
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt

from audio_processing import griffin_lim
from hparams import create_hparams
from model import Tacotron2, load_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence, sequence_to_text

import pdb


# ========== parameters ===========
checkpoint_path = 'models/mellotron_warmup/checkpoint_8500'
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

supportset_sid = '2952'  # m
#supportset_sid = '1069' # f 
output_root = 'audios'

output_dir = os.path.join(
        output_root,
        '-'.join(checkpoint_path.split('/')[-2:]) + '-' + supportset_sid,
)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_mel(path):
    sampling_rate, audio = read(path)
    audio = torch.from_numpy(audio.astype(np.float32))
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
    dataloader = TextMelLoader(audio_path, hparams)
    datacollate = TextMelCollate(1)
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


# save reference wavs and aligned predictions
for idx in range(len(dataloader)):
    audio_path, text, sid = dataloader.audiopaths_and_text[idx]
    if sid != supportset_sid:
        continue

    # save original wav file
    fname_wav = os.path.join(output_dir, 'ref_true_{}.wav'.format(idx))
    copyfile(audio_path, fname_wav)

    text_encoded = torch.LongTensor(\
            text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()
    mel = load_mel(audio_path)


    # save reconstruction from true mel
    fname_wav = os.path.join(output_dir, 'ref_recon_{}.wav'.format(idx))
    with torch.no_grad(): 
        audio = denoiser(waveglow.infer(mel, sigma=0.8), 0.01)[:,0]
    write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
    fname_fig = os.path.join(output_dir, 'ref_mel.png'.format(idx))
    save_figure(mel[0].data.cpu().numpy(), 
            np.zeros((10,10)), fname_fig)
        
    # save parallel prediction
    fname_wav = os.path.join(output_dir, 'ref_parallel_{}.wav'.format(idx))
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(
                (text_encoded, mel, None, None))
        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:,0]
    write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())

    fname_fig = os.path.join(output_dir, 'attention_{}.png'.format(idx))
    save_figure(mel_outputs_postnet[0].data.cpu().numpy(), 
            alignments[0].data.cpu().numpy(), fname_fig)


    print (idx, text)
    break















#
