import argparse
import os
import warnings
import whisper
from typing import List

# Suppress warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Whisper Audio Transcriber CLI")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument("--model", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Model size (default: base)")
    parser.add_argument("--lang", type=str, default=None, help="Language code (e.g., 'ru', 'en'). Auto-detect if not specified.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--format", type=str, nargs="+", default=["txt"], 
                        choices=["txt", "srt", "vtt", "json", "all"],
                        help="Output formats (default: txt)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps). Auto-detect if not specified.")
    return parser.parse_args()

def format_timestamp(seconds: float):
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def format_timestamp_vtt(seconds: float):
    """Converts seconds to VTT timestamp format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

def write_txt(result, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())
    print(f"Saved TXT: {output_path}")

def write_srt(result, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"Saved SRT: {output_path}")

def write_vtt(result, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in result["segments"]:
            start = format_timestamp_vtt(segment["start"])
            end = format_timestamp_vtt(segment["end"])
            text = segment["text"].strip()
            f.write(f"{start} --> {end}\n{text}\n\n")
    print(f"Saved VTT: {output_path}")

def main():
    args = get_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: File not found: {args.audio_path}")
        return

    print(f"Loading model '{args.model}'...")
    model = whisper.load_model(args.model, device=args.device)

    print(f"Transcribing '{args.audio_path}'...")
    # verbose=True prints progress to stdout
    result = model.transcribe(args.audio_path, language=args.lang, verbose=True)

    base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    
    formats = args.format
    if "all" in formats:
        formats = ["txt", "srt", "vtt", "json"]

    for fmt in formats:
        output_filename = f"{base_name}.{fmt}"
        output_path = os.path.join(args.output_dir, output_filename)
        
        if fmt == "txt":
            write_txt(result, output_path)
        elif fmt == "srt":
            write_srt(result, output_path)
        elif fmt == "vtt":
            write_vtt(result, output_path)
        # json is supported natively by result object if needed, but simple dump is easy
        elif fmt == "json":
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved JSON: {output_path}")

    print("Done!")

if __name__ == "__main__":
    main()
