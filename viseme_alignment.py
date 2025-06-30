import os
import json
import yaml
import torch
import numpy as np
import librosa
from torch.optim.lr_scheduler import OneCycleLR
from models_asr import load_ASR_models
from utils import length_to_mask, mask_from_lens, maximum_path
from meldataset import TextCleaner
import time
import re
import phonemizer
from munch import Munch
import argparse
import torchaudio

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def dict_to_dotdict(d):
    if isinstance(d, dict):
        return DotDict({k: dict_to_dotdict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_dotdict(i) for i in d]
    return d

# --- Step 1: Load config & pre-load ASR models (once) ---------------------
# select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# load training/inference config
cfg = dict_to_dotdict(yaml.safe_load(open('Configs/config_ft.yml')))
sr         = cfg.preprocess_params.get('sr', 24000)
hop_length = cfg.preprocess_params.spect_params.get('hop_length', 256)
n_mels     = cfg.preprocess_params.spect_params.get('n_mels', 80)
# print(f"Sample rate: {sr}, hop length: {hop_length}, n_mels: {n_mels}")

# patch torch.load to ignore weights_only flag
_old_torch_load = torch.load
def _torch_load_patch(*args, **kwargs):
    kwargs['weights_only'] = False
    return _old_torch_load(*args, **kwargs)
torch.load = _torch_load_patch

# load ASR & pitch models
with torch.serialization.safe_globals([getattr, OneCycleLR]):
    text_aligner = load_ASR_models(cfg.ASR_path, cfg.ASR_config).to(device).eval()

# restore original loader
torch.load = _old_torch_load

nets = Munch(text_aligner=text_aligner)

nets['text_aligner'] = nets['text_aligner'].to(device)

# record downsampling factor
n_down = nets['text_aligner'].n_down

print("Models loaded and ready.")

# --- Step 2: Load flat global phoneme→viseme map --------------------------
def load_global_map(json_path):
    # with open(json_path, 'r', encoding='utf-8') as f:
    with open(json_path, 'r') as f:
        gm = json.load(f)
    # cast IDs to int
    return {ph: int(vid) for ph, vid in gm.items()}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
    )
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

# --- Step 3: Align audio → phoneme durations ------------------------------
def get_phoneme_durations(audio_arr, phonetic_transcription):
    # 3a) create mel‐spec
    audio = audio_arr
    
    # custom preprocess function
    mel = preprocess(audio)
    mel = mel.squeeze(0)
    ############################
    mel_t = mel.unsqueeze(0).to(device)
    mel_len = torch.tensor([mel_t.shape[-1]], dtype=torch.long)

    # 3b) tokenize phonetic input
    cleaner = TextCleaner()
    if hasattr(cleaner, "text_to_sequence"):
        tok_ids = cleaner.text_to_sequence(phonetic_transcription)
    else:
        tok_ids = [cleaner.word_index_dictionary.get(ch, 0)
                   for ch in phonetic_transcription
                   if ch in cleaner.word_index_dictionary]
        tok_ids.insert(0, 0)
    
    txt_t = torch.tensor(tok_ids, dtype=torch.long).unsqueeze(0).to(device)
    txt_len = torch.tensor([txt_t.shape[-1]], dtype=torch.long)

    # 3c) forward pass → attention & durations
    with torch.no_grad():
        mask = length_to_mask(mel_len // (2**n_down)).to(device)
        _, _, attn = nets['text_aligner'](mel_t, mask, txt_t)
        # drop blank, reshape
        attn = attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
        down_len = mel_len // (2**n_down)
        m2 = mask_from_lens(attn, txt_len.to(device), down_len.to(device))
        path_mat = maximum_path(attn, m2)
        raw_durations = path_mat.sum(axis=-1).cpu()  # shape [1, seq_len]

    # 3d) convert frame counts → seconds
    conv = (hop_length / sr) * (mel_len / down_len)
    secs = raw_durations * conv  # [1, seq_len]

    # 3e) decode token IDs → IPA phonemes
    inv_map = {v: k for k, v in cleaner.word_index_dictionary.items()}
    decoded = [inv_map[t.item()]
               for t in txt_t[0] if t.item() in inv_map]

    return list(zip(decoded, secs[0].tolist()))

# --- Step 4: Merge specials & compute start times -------------------------
def merge_special_marks(phonemes, durations,
                        specials={'ˈ','ː',',','?','.'}):
    merged_ph, merged_du = [], []
    for ph, du in zip(phonemes, durations):
        if ph in specials and merged_du:
            merged_du[-1] += du
        else:
            merged_ph.append(ph)
            merged_du.append(du)
    return merged_ph, merged_du

def compute_starts(durations):
    arr = np.array(durations, dtype=float)
    return np.concatenate(([0.0], arr.cumsum()[:-1]))

# --- Step 5: Generate viseme timeline JSON -------------------------------
def generate_viseme_timeline(audio_arr,
                             phonetic_transcription,
                             global_map_json,
                             default_viseme=0):
    # load flat mapping
    global_map = load_global_map(global_map_json)

    # align → (phoneme, duration_s)
    ph_dur = get_phoneme_durations(audio_arr, phonetic_transcription)
    phs, dus = zip(*ph_dur)
    phs, dus = list(phs), [float(d) for d in dus]

    # merge specials, compute starts
    clean_ph, clean_du = merge_special_marks(phs, dus)
    starts = compute_starts(clean_du)

    # build timeline
    timeline = []
    for t, ph in zip(starts, clean_ph):
        vid = global_map.get(ph, default_viseme)
        timeline.append({
            "offset": round(float(t * 1000), 3),  # in seconds
            "visemeId": int(vid)
        })
    return timeline

def remove_special_characters(text):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    normalized_text = re.sub(chars_to_ignore_regex, '', text).lower()
    return normalized_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate viseme timeline from audio and text."
        )
    parser.add_argument('--audio_file', 
                        default="wav_text_files/11ElevenLabs.mp3", 
                        help="Path to the audio file"
                        )
    parser.add_argument('--text_file', 
                        default="wav_text_files/11ElevenLabs.txt", 
                        help="Path to the text file"
                        )
    parser.add_argument('--global_map_json', 
                        default="global_map.json", 
                        help="Path to the global phoneme-viseme map JSON"
                        )
    parser.add_argument('--output_path', default="wav_text_files", 
                        help="Path to the global phoneme-viseme map JSON"
                        )
    args = parser.parse_args()

    audio_file = args.audio_file
    text_file = args.text_file
    global_map_json = args.global_map_json
    output_path = args.output_path

    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', 
        preserve_punctuation=True,  
        with_stress=False
        )

    print(f"Processing audio file: {audio_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    text = remove_special_characters(text)
    ps = global_phonemizer.phonemize([text])
    phonetic = ps[0]
    print(f"\nInput text: {text}")
    print(f"\nPhonetic transcription: {phonetic}\n")

    audio, _ = librosa.load(audio_file, sr=sr)

    # 2) Warm run for speed
    # print("Warm run timing:")
    # generate_viseme_timeline(audio, phonetic, global_map_json)

    # 3) Generate final timeline.
    start = time.time()
    timeline = generate_viseme_timeline(audio, phonetic, global_map_json)
    print(f"Total time: {((time.time() - start)*1000):.2f} ms.")
    print(f"Audio duration: {len(audio) / sr:.2f} seconds.")

    # save timeline to JSON
    outfile = os.path.join(output_path, audio_file.split('/')[-1][:-4] + "_timeline.json")
    with open(outfile, 'w') as f:
      json.dump(timeline, f, indent=2)

