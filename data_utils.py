import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
import librosa

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict
from yin import compute_yin


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text and speaker ids
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms and f0s from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, speaker_ids=None):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.f0_min = hparams.f0_min
        self.f0_max = hparams.f0_max
        self.harm_thresh = hparams.harm_thresh
        self.p_arpabet = hparams.p_arpabet

        self.cmudict = None
        if hparams.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)

        self.speaker_ids = speaker_ids
        if speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        return d

    def get_f0(self, audio, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
        f0, harmonic_rates, argmins, times = compute_yin(
            audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
            harm_thresh)
        pad = int((frame_length / hop_length) / 2)
        f0 = [0.0] * pad + f0 + [0.0] * pad

        f0 = np.array(f0, dtype=np.float32)
        return f0

    def get_data(self, audiopath_and_text):
        audiopath, text, speaker = audiopath_and_text
        text = self.get_text(text)
        mel, filepath = self.get_mel_and_f0(audiopath)
        speaker_id = self.get_speaker_id(speaker)
        return (text, mel, speaker_id, filepath)

    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_ids[int(speaker_id)]])

    def get_mel_and_f0(self, filepath):
        audio, sampling_rate = load_wav_to_torch(filepath)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec, filepath

    def get_text(self, text):
        text_norm = torch.IntTensor(
            text_to_sequence(text, self.text_cleaners, self.cmudict, self.p_arpabet))

        return text_norm

    def __getitem__(self, index):
        return self.get_data(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        filepaths = []

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            filepaths.append(batch[ids_sorted_decreasing[i]][3])

        model_inputs = (text_padded, input_lengths, mel_padded, gate_padded,
                        output_lengths, speaker_ids, filepaths)

        return model_inputs


class EpisodicLoader(TextMelLoader):

    def __init__(self, audiopaths_and_text, hparams, speaker_dict=None, shuffle=True):

        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.f0_min = hparams.f0_min
        self.f0_max = hparams.f0_max
        self.harm_thresh = hparams.harm_thresh
        self.p_arpabet = hparams.p_arpabet

        self.cmudict = None
        if hparams.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)

        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

        self.nc = hparams.num_common
        self.nq = hparams.num_query
        self.ns = hparams.num_support
        assert self.ns >= self.nc

        _speaker_ids = np.sort(np.unique([x[2] for x in self.audiopaths_and_text]))
        # to map between string format SID and speaker index
        if speaker_dict is None:
            self.speaker_dict = {_speaker_ids[i]: i for i in range(len(_speaker_ids))}
        else:
            self.speaker_dict = speaker_dict

#        # remove sids with small instances
#        remove_keys = []
#        for key, value in d.items():
#            if len(value) < self.ns + self.nq - self.nc:
#                remove_keys.append(key)
#
#        for key in remove_keys:
#            del self.speaker_dict[key]
#            print ('SID {} is removed because of small num. of instances'.format(key))

        new_xys = []
        for i in range(len(self)):
            _, _, sid = self.audiopaths_and_text[i]
            if sid in self.speaker_dict:
                new_xys.append(self.audiopaths_and_text[i])
        self.audiopaths_and_text = new_xys

        d = {} # sid: idxs
        # for keeping instance indexes with the same speaker ids
        for i in range(len(self)):
            _, _, sid = self.audiopaths_and_text[i]
            if sid in self.speaker_dict:
                if sid in d:
                    d[sid].append(i)
                else:
                    d[sid] = [i]

        self.sid_to_index = d

        print ('dataset SIDs: ', self.speaker_dict.keys())
        print ('Num speakers: {}'.format(len(self.speaker_dict)))

    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_dict[speaker_id]])

    def get_mel(self, filepath):
        wav, sampling_rate = load_wav_to_torch(filepath)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = wav / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_mel_and_f0(self, filepath):
        audio, sampling_rate = load_wav_to_torch(filepath)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(
                text_to_sequence(text, self.text_cleaners, self.cmudict, self.p_arpabet)
        )
        return text_norm

    def __getitem__(self, index):
        audiopath, text, speaker = self.audiopaths_and_text[index]
        text = self.get_text(text)
        mel = self.get_mel_and_f0(audiopath)
        speaker_id = self.get_speaker_id(speaker)
        return (text, mel, speaker_id, audiopath)


class EpisodicBatchSampler():
    def __init__(self, iterdict, hparams, shuffle=True):
        # iterdict contains {sid: [idx1, idx2, ...]}
        self.iterdict = iterdict
        self.batch_size = hparams.num_support + hparams.num_query - hparams.num_common
        self.shuffle = shuffle
        self.seed_seed = hparams.seed
        
        count_not_used = self.set_epoch(0)
        print ('Number of excluded instances in 1 epoch: {}'.format(count_not_used))
        

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        np.random.seed(self.seed_seed + epoch)

        batches = []; count_not_used = 0
        for sid in sorted(self.iterdict):
            batch_idxes = self.iterdict[sid]
            if self.shuffle:
                np.random.shuffle(batch_idxes)

            for m in range(len(batch_idxes) // self.batch_size):
                sub_bidx = batch_idxes[m*self.batch_size:(m+1)*self.batch_size]
                batches.append(sub_bidx)
            count_not_used += len(batch_idxes) % self.batch_size

        if self.shuffle:
            np.random.shuffle(batches)

        self.batches = batches
        return count_not_used


class DistributedEpisodicSampler():
    def __init__(self, iterdict, hparams, shuffle=True):

        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

        self.iterdict = iterdict
        self.batch_size = hparams.num_support + hparams.num_query - hparams.num_common
        self.shuffle = shuffle
        self.seed = hparams.seed
        self.shuffle = shuffle
        self.epoch = 0
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch+self.seed)
        batches = []
        for sid in self.iterdict:
            batch_idx = self.iterdict[sid]
            if self.shuffle:
                _rndi = torch.randperm(len(batch_idx), generator=g).tolist()
                batch_idx = [batch_idx[_ri] for _ri in _rndi]

            for m in range(len(batch_idx) // self.batch_size):
                sub_bidx = batch_idx[m*self.batch_size:(m+1)*self.batch_size]
                batches.append(sub_bidx)
        if self.shuffle:
            _rndi = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[_ri] for _ri in _rndi]

        batches = batches[:len(batches) - len(batches) % self.num_replicas]
        assert len(batches) % self.num_replicas == 0
        # only keeps num_replicas times of batches

        subbatches = batches[self.rank:len(batches):self.num_replicas]
        return iter(subbatches)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        size = [len(self.iterdict[sid]) // self.batch_size for sid in self.iterdict]
        return sum(size)

        
class EpisodicCollater():
    def __init__(self, n_frames_per_step, hparams):
        self.n_frames_per_step = n_frames_per_step 
        self.nc = hparams.num_common
        self.nq = hparams.num_query
        self.ns = hparams.num_support
        assert self.ns >= self.nc

    def __call__(self, batch):
        '''
        batch instances shares the same speaker id
        collater function should split query and support set
        each batch contains (text, mel, sid, f0)
        '''
        sids = torch.unique(torch.stack([b[2] for b in batch]))
        assert len(sids) == 1 # accept only sinlge sid 

        support_set_batches = batch[:self.ns]
        query_set_batches = batch[self.ns-self.nc:]
        
        batch = {
            'support': self._subset_collater(support_set_batches),
            'query': self._subset_collater(query_set_batches),
        }
        return batch

    def _subset_collater(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([len(x[0]) for x in batch]),
                dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        datapath = []

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            datapath.append(batch[ids_sorted_decreasing[i]][3])

        outputs = {
            'text_padded': text_padded,
            'input_lengths': input_lengths,
            'mel_padded': mel_padded,
            'gate_padded': gate_padded,
            'output_lengths': output_lengths,
            'speaker_ids': speaker_ids,
            'idx': ids_sorted_decreasing,
            'datapath': datapath,
        }
        
        return outputs
