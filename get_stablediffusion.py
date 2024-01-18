# export HF_ENDPOINT=https://hf-mirror.com
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
snapshot_download(repo_id='stabilityai/stable-diffusion-2-1', cache_dir="./cache", local_dir="./huggingface")

