"""
Copy audio files from mos_server to EMKR demo folder and generate mel-spectrogram images.

Usage:
    pip install librosa matplotlib soundfile
    python copy_and_gen_melspec.py
"""

import os
import shutil
import json
import numpy as np

# ---- Config ----
SRC = r"C:\Users\user\Downloads\mos_server\mos_server\public\audio_sample"
DST = r"C:\Users\user\Desktop\emkr_demo\EMKR"
AUDIO_DIR = os.path.join(DST, "audio")
SPEC_DIR = os.path.join(DST, "spectrograms")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SPEC_DIR, exist_ok=True)

# Task -> list of sample IDs (first 2 per task)
SAMPLES = {
    "add":     ["sample_0017", "sample_0033"],
    "remove":  ["sample_0070", "sample_0083"],
    "replace": ["sample_0002", "sample_0009"],
    "move":    ["sample_0005", "sample_0015"],
    "extend":  ["sample_0008", "sample_0010"],
}

# Models per task
# audit does NOT support move/extend
MODELS_FULL = ["audit", "sao_instruct10", "emkr"]
MODELS_NO_AUDIT = ["sao_instruct10", "emkr"]

MODELS_BY_TASK = {
    "add":     MODELS_FULL,
    "remove":  MODELS_FULL,
    "replace": MODELS_FULL,
    "move":    MODELS_NO_AUDIT,
    "extend":  MODELS_NO_AUDIT,
}

# Model name mapping for output filenames
MODEL_NAMES = {
    "audit": "audit",
    "sao_instruct10": "sao",
    "emkr": "emkr",
}


def gen_melspec(wav_path, out_path, sr=44100):
    """Generate and save a mel-spectrogram image from a wav file."""
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    y, sr = librosa.load(wav_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), dpi=150)
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel",
                             fmax=sr // 2, ax=ax, cmap="magma")
    import matplotlib.ticker as ticker
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.tick_params(axis='x', labelsize=12)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def copy_audio(src_path, dst_path):
    """Copy a wav file if source exists."""
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        return True
    else:
        print(f"  [WARN] Not found: {src_path}")
        return False


def main():
    metadata_all = {}

    for task, sample_ids in SAMPLES.items():
        models = MODELS_BY_TASK[task]
        for idx, sid in enumerate(sample_ids, start=1):
            ex = f"ex{idx}"
            prefix = f"{task}_{ex}"
            print(f"\n--- {prefix} ({sid}) ---")

            emkr_dir = os.path.join(SRC, "emkr", task, sid)

            # 1) Copy & generate melspec for INPUT (from emkr folder)
            input_src = os.path.join(emkr_dir, "input.wav")
            input_dst = os.path.join(AUDIO_DIR, f"{prefix}_input.wav")
            if copy_audio(input_src, input_dst):
                gen_melspec(input_dst, os.path.join(SPEC_DIR, f"{prefix}_input.png"))
                print(f"  input OK")

            # 2) Copy & generate melspec for GT / output (from emkr folder)
            gt_src = os.path.join(emkr_dir, "output.wav")
            gt_dst = os.path.join(AUDIO_DIR, f"{prefix}_gt.wav")
            if copy_audio(gt_src, gt_dst):
                gen_melspec(gt_dst, os.path.join(SPEC_DIR, f"{prefix}_gt.png"))
                print(f"  gt OK")

            # 3) Copy & generate melspec for each model's generated output
            for model in models:
                mname = MODEL_NAMES[model]
                gen_src = os.path.join(SRC, model, task, sid, "generated.wav")
                gen_dst = os.path.join(AUDIO_DIR, f"{prefix}_{mname}.wav")
                if copy_audio(gen_src, gen_dst):
                    gen_melspec(gen_dst, os.path.join(SPEC_DIR, f"{prefix}_{mname}.png"))
                    print(f"  {mname} OK")

            # 4) Read metadata
            meta_path = os.path.join(emkr_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                metadata_all[prefix] = meta
                print(f"  metadata: {meta.get('prompt', '?')}")

    # Save all metadata for reference
    with open(os.path.join(DST, "demo_metadata.json"), "w") as f:
        json.dump(metadata_all, f, indent=2)
    print(f"\nDone! Metadata saved to demo_metadata.json")
    print(f"Audio files: {len(os.listdir(AUDIO_DIR))}")
    print(f"Spectrogram images: {len(os.listdir(SPEC_DIR))}")


if __name__ == "__main__":
    main()
