#!/usr/bin/env python3
# filepath: /home/hyeonuk/medical_rag_multi_agent_framework/huggingface_model_download_aria2.py
"""
사용 전 설치: sudo apt install aria2
"""
import os
import subprocess
from huggingface_hub import HfApi

# 모델 정보
repo_id = "snuh/hari-q2.5-thinking"
local_dir = "./models/hari-q2.5-thinking"

# HuggingFace API로 파일 목록 가져오기
api = HfApi()
files = api.list_repo_files(repo_id)

print(f" Total {len(files)} files to download")
print("Using aria2c for maximum speed...\n")

# 디렉토리 생성
os.makedirs(local_dir, exist_ok=True)

# aria2c로 각 파일 다운로드
base_url = f"https://huggingface.co/{repo_id}/resolve/main"

for file in files:
    url = f"{base_url}/{file}"
    output_path = os.path.join(local_dir, file)
    output_dir = os.path.dirname(output_path)
    
    # 하위 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # aria2c 명령어
    cmd = [
        "aria2c",
        "--max-connection-per-server=16",  # 서버당 최대 연결 수
        "--split=16",                       # 16개로 분할 다운로드
        "--min-split-size=1M",             # 최소 분할 크기
        "--continue=true",                  # 이어받기
        "--dir", output_dir,
        "--out", os.path.basename(output_path),
        url
    ]
    
    print(f" Downloading: {file}")
    subprocess.run(cmd, check=False)

print("\nAll files downloaded!")
print(f" Location: {local_dir}")
