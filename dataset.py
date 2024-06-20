from config import *
import logging
import random
from os.path import join, isfile

import cv2
import librosa
import numpy as np
import torch
from decord import VideoReader
from decord import cpu
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.models.video import S3D_Weights



def subsample_list(inp_list: list, sample_rate: float):
    random.shuffle(inp_list)
    return [inp_list[i] for i in range(int(len(inp_list) * sample_rate))]


class AVSEDataset(Dataset):
    def __init__(self, files_list, shuffle=True, seed=SEED, subsample=1,
                 clipped_batch=True, sample_items=True, time_domain=False):
        super(AVSEDataset, self).__init__()
        self.time_domain = time_domain
        self.clipped_batch = clipped_batch
        self.files_list = files_list
        if shuffle:
            random.seed(SEED)
            random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
        logging.info("Found {} utterances".format(len(self.files_list)))
        self.data_count = len(self.files_list)
        self.batch_index = 0
        self.total_batches_seen = 0
        self.batch_input = {"noisy": None}
        self.index = 0
        self.max_len = len(self.files_list)
        self.max_cache = 0
        self.seed = seed
        self.window = "hann"
        self.fading = False
        self.sample_items = sample_items
        self.video_transform = S3D_Weights.KINETICS400_V1.transforms()

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        while True:
            try:
                data = {}
                if self.sample_items:
                    clean_file, noise_file, noisy_file, mp4_file, scene_id = random.sample(self.files_list, 1)[0]
                else:
                    clean_file, noise_file, noisy_file, mp4_file, scene_id = self.files_list[idx]
                    data["noisy_stft"] = self.get_stft(self.load_wav(noisy_file)).T
                    if isfile(clean_file):
                        data["clean"] = self.load_wav(clean_file)
                    data["scene"] = scene_id

                data["noisy_audio"], clean_audio, data["video_frames"] = self.get_data(clean_file, noise_file,
                                                                                               noisy_file, mp4_file)
                return data, clean_audio
            except Exception as e:
                logging.error("Error in loading data: {}".format(e))

    def load_wav(self, wav_path):
        return wavfile.read(wav_path)[1].astype(np.float32) / (2 ** 15)

    def get_stft(self, audio):
        return librosa.stft(audio, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window="hann",
                            center=True)

    def get_audio_features(self, audio):
        #return np.abs(self.get_stft(audio)).transpose(1, 0).astype(np.float32)
        x = torch.stft(torch.tensor(audio), win_length=window_size, n_fft=stft_size, hop_length=window_shift,
                            return_complex=True, window=torch.hann_window(window_size), normalized=True)#.transpose(1, 2)
        x = torch.view_as_real(x)
        x = x.unsqueeze(0)
        return x

    def get_data(self, clean_file, noise_file, noisy_file, mp4_file):
        noisy = self.load_wav(noisy_file)
        vr = VideoReader(mp4_file, ctx=cpu(0))
        if isfile(clean_file):
            clean = self.load_wav(clean_file)
        else:
            # clean file for test set is not available
            clean = np.zeros(noisy.shape)
        if self.clipped_batch:
            if clean.shape[0] > max_audio_length:
                clip_idx = random.randint(0, clean.shape[0] - max_audio_length)
                video_idx = int((clip_idx / 16000) * 25)
                clean = clean[clip_idx:clip_idx + max_audio_length]
                noisy = noisy[clip_idx:clip_idx + max_audio_length]
            else:
                video_idx = -1
                clean = np.pad(clean, pad_width=[0, max_audio_length - clean.shape[0]], mode="constant")
                noisy = np.pad(noisy, pad_width=[0, max_audio_length - noisy.shape[0]], mode="constant")
            if len(vr) < max_video_length:
                frames = vr.get_batch(list(range(len(vr)))).asnumpy()
            else:
                max_idx = min(video_idx + max_video_length, len(vr))
                frames = vr.get_batch(list(range(video_idx, max_idx))).asnumpy()
            #bg_frames = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]
            bg_frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
            bg_frames = self.video_transform(bg_frames)
            #bg_frames = np.array([cv2.resize(bg_frames[i], video_frame_size) for i in range(len(bg_frames))]).astype(
            #    np.float32)
            #bg_frames /= 255.0
            if bg_frames.shape[1] < max_video_length:
                bg_frames = torch.cat([bg_frames, torch.zeros(bg_frames.shape[0], max_video_length - bg_frames.shape[1],
                                                               bg_frames.shape[2], bg_frames.shape[3])], dim=1)
        else:
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()
            #bg_frames = np.array(
            #    [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]).astype(np.float32)
            #bg_frames = np.array([cv2.resize(bg_frames[i], video_frame_size) for i in range(len(bg_frames))]).astype(
            #    np.float32)
            #bg_frames /= 255.0
            bg_frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
            bg_frames = self.video_transform(bg_frames)
        if self.time_domain:
            return noisy, clean, bg_frames#[..., np.newaxis]
        
        #bg_frames = self.video_transform(bg_frames)
        return self.get_audio_features(noisy), self.get_audio_features(clean), bg_frames


class AVSEChallengeDataModule:
    def __init__(self, data_root, batch_size=4, time_domain=False):
        super(AVSEChallengeDataModule, self).__init__()
        self.train_dataset_batch = AVSEDataset(self.get_files_list(join(data_root, "train")), time_domain=time_domain)
        self.dev_dataset_batch = AVSEDataset(self.get_files_list(join(data_root, "dev")), time_domain=time_domain)
        self.dev_dataset = AVSEDataset(self.get_files_list(join(data_root, "dev")),
                                       clipped_batch=False, sample_items=False, time_domain=time_domain)
        # !TODO Uncomment this for test set
        self.test_dataset = AVSEDataset(self.get_files_list(join(data_root, "eval"), test_set=True), sample_items=False,
                                         clipped_batch=False, time_domain=time_domain)
        self.batch_size = batch_size

    @staticmethod
    def get_files_list(data_root, test_set=False):
        files_list = []
        for file in os.listdir(join(data_root, "scenes")):
            if not test_set:
                if file.endswith("mixed.wav"):
                    files = (join(data_root, "scenes", file.replace("mixed", "target")),
                            join(data_root, "scenes", file.replace("mixed", "interferer")),
                            join(data_root, "scenes", file),
                            join(data_root, "scenes", file.replace("_mixed.wav", "_silent.mp4")),
                            file.replace("_mixed.wav", "")
                            )
                    if not test_set:
                        if all([isfile(f) for f in files[:-1]]):
                            files_list.append(files)
                    else:
                        files_list.append(files)
            else:
                if file.endswith("mixed.wav"):
                    files = (
                        "",
                        "",
                        join(data_root, "scenes", file),
                        join(data_root, "scenes", file.replace("_mixed.wav", "_silent.mp4")),
                        file.replace("_mixed.wav", "")
                    )
                    files_list.append(files)
        return files_list

    def train_dataloader(self):
        assert len(self.train_dataset_batch) > 0, "No training data found"
        return torch.utils.data.DataLoader(self.train_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        assert len(self.dev_dataset_batch) > 0, "No validation data found"
        return torch.utils.data.DataLoader(self.dev_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True,
                                           persistent_workers=True)

if __name__ == '__main__':
    data_module = AVSEChallengeDataModule(data_root="/home/jesus10/jrenato/avse/data/")
    #data_module.train_dataset_batch.get_data(*data_module.train_dataset_batch.files_list[0])
    train_loader = data_module.train_dataloader()
    for batch in tqdm(train_loader):
        print("Noisy audio shape: ", batch[0]["noisy_audio"].shape)
        print("Clean audio shape: ", batch[1].shape)
        print("Video frames shape: ", batch[0]["video_frames"].shape)
        # pass
    dev_loader = data_module.val_dataloader()
    for batch in tqdm(dev_loader):
        pass
    # test_loader = data_module.test_dataloader()
    # for batch in tqdm(test_loader):
    #     pass
    print("Data loading successful")