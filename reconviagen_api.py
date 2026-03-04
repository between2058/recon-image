import os
import shutil
import tempfile
import uuid
import gc
import asyncio
import datetime
import logging
import logging.handlers
import torch
import numpy as np
import imageio
from PIL import Image
from typing import Literal

# --- 1. 環境變數設定 (必須在所有 import 之前) ---
os.environ['SPCONV_ALGO'] = 'native'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

# --- ReconViaGen 特有的 Imports ---
from trellis.pipelines import TrellisVGGTTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# =============================================================================
# Logging 設定
# =============================================================================

os.makedirs("/app/logs", exist_ok=True)


class TaiwanFormatter(logging.Formatter):
    """自訂 Formatter，固定使用 UTC+8 (台灣時間)，不依賴系統時區。"""
    _TZ = datetime.timezone(datetime.timedelta(hours=8))

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=self._TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{record.msecs:03.0f}"


class HealthCheckFilter(logging.Filter):
    """過濾 uvicorn.access 裡的 GET /health，防止 log 刷屏。"""
    def filter(self, record):
        return "GET /health" not in record.getMessage()


def _rotating_file_handler(filename: str, formatter: logging.Formatter) -> logging.Handler:
    handler = logging.handlers.TimedRotatingFileHandler(
        f"/app/logs/{filename}",
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


# Formatters
_fmt = TaiwanFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_access_fmt = TaiwanFormatter("%(asctime)s %(message)s")

# app logger — 業務邏輯，同時輸出到 stdout + app.log
app_logger = logging.getLogger("app")
app_logger.setLevel(logging.DEBUG)
app_logger.propagate = False
app_logger.addHandler(_rotating_file_handler("app.log", _fmt))
app_logger.addHandler(logging.StreamHandler())

# uvicorn.access logger — 過濾 /health + 寫入 access.log
_uvicorn_access = logging.getLogger("uvicorn.access")
_uvicorn_access.addFilter(HealthCheckFilter())
_uvicorn_access.addHandler(_rotating_file_handler("access.log", _access_fmt))

# uvicorn logger — server 啟動/錯誤，寫入 uvicorn.log
_uvicorn = logging.getLogger("uvicorn")
_uvicorn.addHandler(_rotating_file_handler("uvicorn.log", _fmt))

# =============================================================================

VALID_MULTIIMAGE_ALGOS = {'multidiffusion', 'stochastic'}

app = FastAPI(title="ReconViaGen API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = tempfile.mkdtemp()
app_logger.info(f"Output directory: {OUTPUT_DIR}")

# --- 全局變數 ---
pipeline = None
gpu_lock = asyncio.Lock()


# =============================================================================
# GPU 記憶體追蹤
# =============================================================================

def log_gpu_memory(label: str):
    """印出目前 GPU 已分配與保留的記憶體量。"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        app_logger.info(
            f"GPU memory [{label}]: allocated={allocated:.2f} GB  reserved={reserved:.2f} GB"
        )


# =============================================================================
# Model Loading
# =============================================================================

def ensure_model_loaded():
    """
    檢查模型是否已載入，若未載入則執行載入動作。
    此函式應在 gpu_lock 保護下執行。
    """
    global pipeline
    if pipeline is None:
        app_logger.info("[Lazy Load] 偵測到模型尚未載入，正在初始化 ReconViaGen 模型...")
        log_gpu_memory("before model load")
        try:
            loaded_pipeline = TrellisVGGTTo3DPipeline.from_pretrained("esther11/trellis-vggt-v0-2")
            loaded_pipeline.cuda()
            loaded_pipeline.VGGT_model.cuda()
            loaded_pipeline.birefnet_model.cuda()

            pipeline = loaded_pipeline
            log_gpu_memory("model loaded")
            app_logger.info("[Lazy Load] 模型載入完成！")
        except Exception as e:
            app_logger.error(f"模型載入失敗: {e}")
            raise RuntimeError(f"Model loading failed: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "gpu_busy": gpu_lock.locked(),
    }


@app.post("/generate-single")
async def generate_single_image(
    file: UploadFile = File(...),
    seed: int = 0,
    simplify: float = 0.95,
    texture_size: int = 1024,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 30,
    slat_guidance_strength: float = 3.0,
    slat_sampling_steps: int = 12,
):
    try:
        request_id = str(uuid.uuid4())
        req_dir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(req_dir, exist_ok=True)

        input_path = os.path.join(req_dir, "input.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image = Image.open(input_path)

        return await _run_pipeline(
            images=[image],
            request_id=request_id,
            seed=seed,
            simplify=simplify,
            texture_size=texture_size,
            ss_guidance_strength=ss_guidance_strength,
            ss_sampling_steps=ss_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            slat_sampling_steps=slat_sampling_steps,
            multiimage_algo="stochastic"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-batch")
async def generate_batch_images(
    files: list[UploadFile] = File(...),
    seed: int = 0,
    simplify: float = 0.95,
    texture_size: int = 1024,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 30,
    slat_guidance_strength: float = 3.0,
    slat_sampling_steps: int = 12,
):
    """
    Batch Mode:
    接收多張不同圖片，依序對每張圖片執行 Single Image Inference。

    HTTP status:
      200 — all items succeeded
      207 — partial success (some failed, some succeeded)
      400 — no files uploaded
      500 — all items failed
    """
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="請至少上傳一張圖片")

    batch_results = []
    app_logger.info(f"[Batch Start] 收到 {len(files)} 張圖片，準備排隊處理...")

    try:
        for i, file in enumerate(files):
            request_id = f"{uuid.uuid4()}_batch_{i}"
            req_dir = os.path.join(OUTPUT_DIR, request_id)
            os.makedirs(req_dir, exist_ok=True)

            app_logger.info(f"  [{i+1}/{len(files)}] 正在處理: {file.filename}")

            input_path = os.path.join(req_dir, "input.png")
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            image = Image.open(input_path)

            try:
                result = await _run_pipeline(
                    images=[image],
                    request_id=request_id,
                    seed=seed,
                    simplify=simplify,
                    texture_size=texture_size,
                    ss_guidance_strength=ss_guidance_strength,
                    ss_sampling_steps=ss_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    slat_sampling_steps=slat_sampling_steps,
                    multiimage_algo="stochastic"
                )

                result["original_filename"] = file.filename
                result["status"] = "success"
                batch_results.append(result)

            except Exception as e:
                app_logger.error(f"  [{file.filename}] 處理失敗: {e}")
                batch_results.append({
                    "original_filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })

        app_logger.info("[Batch End] 所有任務處理完成。")

        succeeded = sum(1 for r in batch_results if r.get("status") == "success")
        failed = len(batch_results) - succeeded

        response_body = {
            "total_count": len(files),
            "succeeded": succeeded,
            "failed": failed,
            "results": batch_results,
        }

        if succeeded == 0:
            raise HTTPException(
                status_code=500,
                detail=f"All {len(files)} items failed. See 'results' for per-item errors.",
            )

        from fastapi.responses import JSONResponse
        status_code = 207 if failed > 0 else 200
        return JSONResponse(content=response_body, status_code=status_code)

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Batch Critical Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-multi")
async def generate_multi_image(
    files: list[UploadFile] = File(...),
    seed: int = 0,
    simplify: float = 0.95,
    texture_size: int = 1024,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 30,
    slat_guidance_strength: float = 3.0,
    slat_sampling_steps: int = 12,
    multiimage_algo: str = "multidiffusion"
):
    if multiimage_algo not in VALID_MULTIIMAGE_ALGOS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid multiimage_algo '{multiimage_algo}'. Must be one of: {sorted(VALID_MULTIIMAGE_ALGOS)}",
        )

    try:
        if len(files) < 1:
            raise HTTPException(status_code=400, detail="請至少上傳一張圖片")

        request_id = str(uuid.uuid4())
        req_dir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(req_dir, exist_ok=True)

        loaded_images = []
        for i, file in enumerate(files):
            file_path = os.path.join(req_dir, f"input_{i}.png")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            loaded_images.append(Image.open(file_path))

        return await _run_pipeline(
            images=loaded_images,
            request_id=request_id,
            seed=seed,
            simplify=simplify,
            texture_size=texture_size,
            ss_guidance_strength=ss_guidance_strength,
            ss_sampling_steps=ss_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            slat_sampling_steps=slat_sampling_steps,
            multiimage_algo=multiimage_algo
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_pipeline(
    images, request_id, seed, simplify, texture_size,
    ss_guidance_strength, ss_sampling_steps,
    slat_guidance_strength, slat_sampling_steps,
    multiimage_algo
):
    async with gpu_lock:
        await run_in_threadpool(ensure_model_loaded)

        app_logger.info(f"開始處理請求 {request_id}")
        output_dir = os.path.join(OUTPUT_DIR, request_id)

        def run_inference():
            global pipeline

            app_logger.info(f"[Step 1] 正在執行去背與預處理... (數量: {len(images)})")
            processed_images = []
            for img in images:
                img = img.convert("RGBA")
                proc_img = pipeline.preprocess_image(img)
                processed_images.append(proc_img)

            app_logger.info(f"預處理完成，圖片尺寸: {processed_images[0].size}")

            app_logger.info("[Step 2] 開始 3D 生成...")
            return pipeline.run(
                image=processed_images,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
                mode=multiimage_algo,
            )

        try:
            outputs, _, _ = await run_in_threadpool(run_inference)

            video_gs = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
            video_gs_path = os.path.join(output_dir, "gs.mp4")
            imageio.mimsave(video_gs_path, video_gs, fps=15)

            video_mesh = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
            video_mesh_path = os.path.join(output_dir, "mesh.mp4")
            imageio.mimsave(video_mesh_path, video_mesh, fps=15)

            video_rf_path = os.path.join(output_dir, "rf.mp4")
            shutil.copy(video_gs_path, video_rf_path)

            gs = outputs['gaussian'][0]
            mesh = outputs['mesh'][0]
            glb = postprocessing_utils.to_glb(gs, mesh, simplify=simplify, texture_size=texture_size, verbose=False)
            glb_path = os.path.join(output_dir, "output.glb")
            glb.export(glb_path)

            ply_path = os.path.join(output_dir, "output.ply")
            gs.save_ply(ply_path)

            torch.cuda.empty_cache()

            return {
                "gaussian_video": f"/download/{request_id}/gs.mp4",
                "radiance_video": f"/download/{request_id}/rf.mp4",
                "mesh_video": f"/download/{request_id}/mesh.mp4",
                "glb_file": f"/download/{request_id}/output.glb",
                "ply_file": f"/download/{request_id}/output.ply"
            }

        except Exception as e:
            app_logger.error(f"Inference Error: {e}")
            raise e

        finally:
            # global pipeline
            # del pipeline
            # pipeline = None
            # gc.collect()
            torch.cuda.empty_cache()
            log_gpu_memory("after flush")
            # app_logger.info("模型已卸載")


@app.get("/download/{request_id}/{file_name}")
async def download_file(request_id: str, file_name: str):
    file_path = os.path.join(OUTPUT_DIR, request_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="找不到文件")

    media_type = 'application/octet-stream'
    if file_name.endswith('.mp4'):
        media_type = 'video/mp4'
    elif file_name.endswith('.glb'):
        media_type = 'model/gltf-binary'

    return FileResponse(file_path, media_type=media_type, filename=file_name)


@app.on_event("shutdown")
async def cleanup():
    try:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=52069)
