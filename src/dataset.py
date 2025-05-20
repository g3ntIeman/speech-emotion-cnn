import os
import glob
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch.nn as nn

# Парсинг имени файла

def parse_filename(filename):
    parts = filename.split("-")
    return {
        "modality": int(parts[0]),
        "vocal_channel": int(parts[1]),
        "emotion": int(parts[2]),
        "intensity": int(parts[3]),
        "statement": int(parts[4]),
        "repetition": int(parts[5]),
        "actor": int(parts[6].split(".")[0]),
    }

# Загрузка метаданных

def load_metadata(audio_dir):
    files = glob.glob(os.path.join(audio_dir, "Actor_*", "*.wav"))
    print("Файлов найдено:", len(files))
    data = []
    for file in files:
        meta = parse_filename(os.path.basename(file))
        meta["file"] = file
        data.append(meta)
    df = pd.DataFrame(data)
    df["emotion_label"] = df["emotion"].map({
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
    })
    return df

# Улучшенная трансформация спектрограммы
transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=48000, n_mels=32),
    T.AmplitudeToDB()
)

def wrapped_transform(waveform, max_len=800):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    melspec = transform(waveform)

    if melspec.shape[-1] < max_len:
        pad_amount = max_len - melspec.shape[-1]
        melspec = torch.nn.functional.pad(melspec, (0, pad_amount))
    else:
        melspec = melspec[:, :, :max_len]

    return melspec  # [1, 32, 800]

# Кастомный датасет
class EmotionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {label: i for i, label in enumerate(sorted(df['emotion_label'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df.loc[idx, 'file']
        label = self.label_map[self.df.loc[idx, 'emotion_label']]
        waveform, sample_rate = torchaudio.load(file_path)
        if self.transform:
            features = self.transform(waveform)
        else:
            features = waveform
        return features, label