import os
import sys
from pathlib import Path
import urllib.request
import time

MODELS_DIR = Path("C:/Projects/Mark/models/kokoro")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

def download_file(filename, url):
    target = MODELS_DIR / filename
    if target.exists() and target.stat().st_size > 1000:
        print(f"[Kokoro Download] {filename} already exists ({target.stat().st_size} bytes). Skipping.")
        return

    print(f"[Kokoro Download] Downloading {filename} from {url}...")
    temp_target = target.with_suffix(".tmp")
    
    start_time = time.time()
    def reporthook(count, block_size, total_size):
        if total_size > 0 and count % 500 == 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"  {filename}: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", flush=True)

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, temp_target, reporthook)
    
    temp_target.rename(target)
    duration = time.time() - start_time
    print(f"[Kokoro Download] {filename} completed in {duration:.1f}s ({target.stat().st_size / (1024*1024):.1f} MB).")

if __name__ == "__main__":
    for fname, furl in FILES.items():
        download_file(fname, furl)
    print("[Kokoro Download] All model files ready!")
