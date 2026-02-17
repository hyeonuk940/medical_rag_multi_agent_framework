import os
from huggingface_hub import snapshot_download

# 환경 변수 설정 (스크립트 내부에서 명시)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

snapshot_download(
    repo_id="snuh/hari-q2.5-thinking",
    local_dir="./models/hari-q2.5-thinking",
    max_workers=64,           # 워커 수를 64개까지 대폭 증폭 (OMEN 사양 믿고 가기)
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5"], # 불필요한 가중치 포맷 제외 (필요시)
)