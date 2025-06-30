# ASRCNN-PL-alignment

```sh
git clone https://github.com/Rehan-Ahmad/ASRCNN-PL-alignment.git
```

This repository provides a tool for generating viseme timelines from audio and text using ASR models and phoneme-to-viseme mapping. It is designed for applications such as lip-syncing and speech animation.

## Requirements

Install dependencies with:

```sh
pip install SoundFile torchaudio munch torch pydub pyyaml librosa nltk matplotlib accelerate transformers phonemizer einops einops-exts tqdm typing-extensions git+https://github.com/resemble-ai/monotonic_align.git
sudo apt-get install espeak-ng
```

## Usage

### Command Line

```sh
python viseme_alignment.py --audio_file path/to/audio.mp3 --text_file path/to/text.txt --global_map_json path/to/global_map.json --output_path output/directory
```

- `--audio_file`: Path to the input audio file (default: `wav_text_files/11ElevenLabs.mp3`)
- `--text_file`: Path to the input text file (default: `wav_text_files/11ElevenLabs.txt`)
- `--global_map_json`: Path to the phoneme-viseme mapping JSON (default: `global_map.json`)
- `--output_path`: Directory to save the output timeline JSON (default: `wav_text_files`)

## Configuration

Model and preprocessing parameters are loaded from `Configs/config_ft.yml`. Adjust this file to match your setup.

## Notes

- The script expects the ASR model and config files to be specified in `Configs/config_ft.yml`.
