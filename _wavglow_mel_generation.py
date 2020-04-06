import argparse
import numpy as np
import math
import torch
from model import load_model
from data_utils import TextMelLoader, TextMelCollate
from torch.utils.data import DataLoader
from hparams import create_hparams
from tqdm import tqdm

import pdb
# no distributed supported

def prepare_dataloaders(hparams):

    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    train_sampler = None
    shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler


def run(output_dir, ckpt_path):

    model = load_model(hparams)
    checkpoint_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])

    train_loader, valset, collate_fn, train_sampler = prepare_dataloaders(hparams)

    model.eval()
    for batch in tqdm(train_loader):

        text, _, mel, _, _, _, fname = batch
        mel_pred, attn = model.inference((text.cuda(), mel.cuda()))
        
        output_fname = fname[0].replace('.wav', '-kkr2.mel')
        mel = mel_pred[0].data.cpu().numpy()
        np.save(output_fname, mel)


if __name__ == '__main__':

    output_dir = 'data-bin/mel_train-clean-100'
    ckpt_path = 'models/gst_tacotron_baseline_pretrained/checkpoint_45000'
    
    hparams = create_hparams()
    hparams.batch_size = 1
    run(output_dir, ckpt_path)
