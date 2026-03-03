# ReconViaGen API — Deployment Guide

部署目標：NVIDIA RTX Pro 6000（Blackwell, sm_120）、CUDA 12.8、Ubuntu Server

---

## 目錄

1. [前置需求](#1-前置需求)
2. [複製專案到 Server](#2-複製專案到-server)
3. [Build Docker Image](#3-build-docker-image)
4. [啟動服務](#4-啟動服務)
5. [確認服務正常](#5-確認服務正常)
6. [API 使用方式](#6-api-使用方式)
7. [常用維運指令](#7-常用維運指令)
8. [排錯指南](#8-排錯指南)

---

## 1. 前置需求

在目標 Server 上依序確認以下項目：

### 1.1 NVIDIA Driver

```bash
nvidia-smi
```

確認輸出中包含 `CUDA Version: 12.8`，且 GPU 顯示為 RTX Pro 6000。

> Driver 版本需 **≥ 570**（支援 CUDA 12.8 / sm_120 的最低需求）。

### 1.2 Docker Engine

```bash
docker --version   # 需 20.10+
docker compose version  # 需 2.x（V2 語法）
```

若尚未安裝：

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### 1.3 NVIDIA Container Toolkit

讓 Docker 容器能存取 GPU：

```bash
# 安裝
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 設定並重啟 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 驗證
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### 1.4 確認 Host 路徑存在

```bash
ls /pegaai/model_team/huggingface_cache
```

若目錄不存在，先建立：

```bash
sudo mkdir -p /pegaai/model_team/huggingface_cache
```

---

## 2. 複製專案到 Server

```bash
# 從你的開發機推到 Server（擇一）

# 方法 A：git clone（推薦）
git clone <your-repo-url> /home/$USER/ReconViaGen
cd /home/$USER/ReconViaGen

# 方法 B：直接 scp
scp -r ./ReconViaGen user@server-ip:/home/user/ReconViaGen
```

---

## 3. Build Docker Image

> **首次 Build 約需 40–90 分鐘**，主要耗時在 `flash-attn` 從 source 編譯 sm_120 CUDA kernel。

```bash
cd /home/$USER/ReconViaGen
```

### 選項 A：只針對 RTX Pro 6000（速度較快）

```bash
docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="12.0" \
  --build-arg MAX_JOBS="$(nproc)" \
  -t reconviagen:latest .
```

### 選項 B：多架構通用（可移植到其他 GPU）

```bash
docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
  --build-arg MAX_JOBS="$(nproc)" \
  -t reconviagen:latest .
```

> `MAX_JOBS` 控制 ninja 的平行編譯執行緒數。設定為 CPU 核心數可加快速度，但 RAM 用量也會等比增加。若 Build 中途 OOM，改小這個數字（例如 `MAX_JOBS=4`）。

---

## 4. 啟動服務

```bash
docker compose up -d
```

這會：
- 掛載 `/pegaai:/notebooks:rw`（含 HuggingFace model cache）
- 將 Port `52069` 暴露到 host
- 設定 `HF_HOME=/notebooks/model_team/huggingface_cache`
- 在背景持續執行（`restart: unless-stopped`）

**第一次啟動後，發出第一個 API 請求時**，服務才會觸發 lazy load，從 HuggingFace 下載模型 `esther11/trellis-vggt-v0-2`。下載完成後，模型會存在 `/pegaai/model_team/huggingface_cache`，之後重啟不需再下載。

---

## 5. 確認服務正常

### 5.1 查看 Log

```bash
docker compose logs -f reconviagen
```

服務啟動成功時會看到：

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:52069
```

### 5.2 Health Check

```bash
curl http://localhost:52069/health
```

回應範例（模型尚未載入）：

```json
{
  "status": "ok",
  "model_loaded": false,
  "gpu_busy": false
}
```

回應範例（模型已載入，服務就緒）：

```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_busy": false
}
```

### 5.3 確認 GPU 可見

```bash
docker compose exec reconviagen nvidia-smi
```

---

## 6. API 使用方式

Base URL：`http://<server-ip>:52069`

### 6.1 單張圖片生成 3D

```bash
curl -X POST http://localhost:52069/generate-single \
  -F "file=@your_image.png" \
  -F "seed=42" \
  -F "simplify=0.95" \
  -F "texture_size=1024" \
  -F "ss_guidance_strength=7.5" \
  -F "ss_sampling_steps=30" \
  -F "slat_guidance_strength=3.0" \
  -F "slat_sampling_steps=12"
```

### 6.2 多視角圖片融合生成 3D

```bash
curl -X POST http://localhost:52069/generate-multi \
  -F "files=@view1.png" \
  -F "files=@view2.png" \
  -F "files=@view3.png" \
  -F "multiimage_algo=multidiffusion"
```

`multiimage_algo` 可選：`multidiffusion`（預設）或 `stochastic`。

### 6.3 批次處理（多張獨立圖片）

```bash
curl -X POST http://localhost:52069/generate-batch \
  -F "files=@obj1.png" \
  -F "files=@obj2.png" \
  -F "files=@obj3.png"
```

### 6.4 下載輸出檔案

API 回傳的路徑格式為 `/download/{request_id}/{filename}`：

```bash
# 範例回應：
# { "glb_file": "/download/abc123/output.glb", ... }

curl -O http://localhost:52069/download/abc123/output.glb
curl -O http://localhost:52069/download/abc123/output.ply
curl -O http://localhost:52069/download/abc123/gs.mp4
curl -O http://localhost:52069/download/abc123/mesh.mp4
```

### 6.5 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `seed` | `0` | 隨機種子，固定可重現結果 |
| `simplify` | `0.95` | Mesh 簡化率（0–1），越小面數越少 |
| `texture_size` | `1024` | Texture 解析度（256 / 512 / 1024 / 2048）|
| `ss_guidance_strength` | `7.5` | Sparse structure CFG 強度 |
| `ss_sampling_steps` | `30` | Sparse structure 推理步數 |
| `slat_guidance_strength` | `3.0` | SLAT CFG 強度 |
| `slat_sampling_steps` | `12` | SLAT 推理步數 |

---

## 7. 常用維運指令

```bash
# 查看 log（即時）
docker compose logs -f reconviagen

# 重啟服務
docker compose restart reconviagen

# 停止服務
docker compose down

# 進入容器 shell（debug 用）
docker compose exec reconviagen bash

# 查看 GPU 記憶體使用
docker compose exec reconviagen nvidia-smi

# 查看容器資源使用
docker stats reconviagen-api

# 更新程式碼後重新 build 並部署
docker compose up -d --build
```

---

## 8. 排錯指南

### GPU 找不到

```
Error: CUDA not available
```

確認 NVIDIA Container Toolkit 安裝正確，並重啟 Docker daemon：

```bash
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

### spconv 在 sm_120 上 Kernel 錯誤

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

`spconv-cu120` 的 prebuilt wheel 未包含 sm_120 kernel。需改為從 source 編譯。
在 `Dockerfile` Step 4 改為：

```dockerfile
RUN pip install --no-cache-dir \
    "git+https://github.com/traveller59/spconv.git"
```

然後重新 build：

```bash
docker build --no-cache -t reconviagen:latest .
```

---

### flash-attn Build 時 OOM

```
c++: fatal error: Killed signal terminated program cc1plus
```

降低平行度：

```bash
docker build --build-arg MAX_JOBS=2 -t reconviagen:latest .
```

---

### HuggingFace 下載失敗 / 403

若模型為 gated repo，需提供 Token：

```bash
# 在 docker-compose.yml 的 environment 加上：
- HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxx
```

或直接在 `docker compose up` 前設置環境變數：

```bash
HF_TOKEN=hf_xxxx docker compose up -d
```

---

### /pegaai 掛載後權限不足

```
PermissionError: [Errno 13] Permission denied: '/notebooks/model_team/...'
```

確認 container 能寫入 host 目錄：

```bash
# 在 host 上
sudo chmod -R 777 /pegaai/model_team/huggingface_cache
# 或指定 Docker 跑的 user id
sudo chown -R $UID:$GID /pegaai/model_team/huggingface_cache
```

---

### 服務回應 500 但 GPU 正常

查看詳細錯誤：

```bash
docker compose logs reconviagen | tail -50
```

常見原因：模型 lazy load 失敗（網路問題、磁碟空間不足）。確認 `/pegaai` 有足夠空間：

```bash
df -h /pegaai
```
