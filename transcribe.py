import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

parser = argparse.ArgumentParser(description="Transcribe an audio file.")
parser.add_argument("source_file", type=Path, help="Audio file to transcribe")
parser.add_argument("output_folder", type=Path, help="Folder for generated files")
args = parser.parse_args()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    parser.error("set HF_TOKEN in the .env file beside the script")

if not args.source_file.is_file():
    parser.error(f"source file not found: {args.source_file}")

args.output_folder.mkdir(parents=True, exist_ok=True)

import whisperx
import torch
import gc

# --- NVIDIA GPU CONFIGURATION (12 GB) ---
audio_file = str(args.source_file)
whisper_model = "large-v3"         # Highest accuracy (fits comfortably in 12 GB)
device = "cuda"
compute_type = "float16"

# Check that the GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("NVIDIA GPU (CUDA) not detected! Check your PyTorch installation.")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")

print("1/3 - Transcribing audio with WhisperX (large-v3) on the GPU...")
model = whisperx.load_model(whisper_model, device, compute_type=compute_type, language="pt")
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=16)

# Free GPU VRAM
del model
gc.collect()
torch.cuda.empty_cache()

print("2/3 - Aligning word timestamps on the GPU...")
model_a, metadata = whisperx.load_align_model(language_code="pt", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device)

del model_a
gc.collect()
torch.cuda.empty_cache()

print("3/3 - Diarizing speakers on the GPU...")
from whisperx.diarize import DiarizationPipeline

diarize_model = DiarizationPipeline(token=hf_token, device=device)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

del diarize_model
gc.collect()
torch.cuda.empty_cache()

# Build the transcript with identified speakers
dialogue = ""
for seg in result["segments"]:
    speaker = seg.get("speaker", "ORADOR_DESCONHECIDO")
    text = seg["text"].strip()
    dialogue += f"[{speaker}]: {text}\n"

output_file = args.output_folder / f"{args.source_file.stem}.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(dialogue)

print("\nTranscription complete! Generated file:")
print(output_file)
