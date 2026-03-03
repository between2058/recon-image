import os
import shutil
import tempfile
import uuid
import gc
import asyncio
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

VALID_MULTIIMAGE_ALGOS = {'multidiffusion', 'stochastic'}

app = FastAPI(title="ReconViaGen API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = tempfile.mkdtemp()
print(f"Output directory: {OUTPUT_DIR}")

# --- 全局變數 ---
pipeline = None
gpu_lock = asyncio.Lock()

# --- [修改] 移除原本的 startup event，改為自定義載入函式 ---
def ensure_model_loaded():
    """
    檢查模型是否已載入，若未載入則執行載入動作。
    此函式應在 gpu_lock 保護下執行。
    """
    global pipeline
    if pipeline is None:
        print("⏳ [Lazy Load] 偵測到模型尚未載入，正在初始化 ReconViaGen 模型...")
        try:
            # 載入模型
            loaded_pipeline = TrellisVGGTTo3DPipeline.from_pretrained("esther11/trellis-vggt-v0-2")
            loaded_pipeline.cuda()
            loaded_pipeline.VGGT_model.cuda()
            loaded_pipeline.birefnet_model.cuda()
            
            pipeline = loaded_pipeline
            print("✅ [Lazy Load] 模型載入完成！")
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    else:
        # 模型已經在記憶體中，直接略過
        pass

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
    print(f"📦 [Batch Start] 收到 {len(files)} 張圖片，準備排隊處理...")

    try:
        for i, file in enumerate(files):
            # 1. 準備獨立的 ID 與目錄
            request_id = f"{uuid.uuid4()}_batch_{i}"
            req_dir = os.path.join(OUTPUT_DIR, request_id)
            os.makedirs(req_dir, exist_ok=True)

            print(f"  👉 [{i+1}/{len(files)}] 正在處理: {file.filename}")

            # 2. 儲存並讀取圖片
            input_path = os.path.join(req_dir, "input.png")
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            image = Image.open(input_path)

            # 3. 呼叫現有的 Pipeline
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
                print(f"  ❌ [{file.filename}] 處理失敗: {e}")
                batch_results.append({
                    "original_filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })

        print(f"✅ [Batch End] 所有任務處理完成。")

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
        print(f"❌ Batch Critical Error: {e}")
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
    # 這裡使用 Lock，確保同時間只有一個請求在載入或執行模型
    async with gpu_lock:
        
        # --- [修改] 在進入鎖之後，執行推理之前，確保模型已載入 ---
        # 這樣做可以實現 Lazy Loading (第一次請求時才載入)
        # 如果你希望每次跑完都釋放記憶體，可以在 try-finally 區塊做 unload
        await run_in_threadpool(ensure_model_loaded)

        print(f"🚀 開始處理請求 {request_id}")
        output_dir = os.path.join(OUTPUT_DIR, request_id)

        def run_inference():
            # 注意：這裡使用 global pipeline，它已經在 ensure_model_loaded 中被賦值了
            global pipeline
            
            # 1. 顯式執行預處理
            print(f"🔄 [Step 1] 正在執行去背與預處理... (數量: {len(images)})")
            processed_images = []
            for img in images:
                img = img.convert("RGBA")
                proc_img = pipeline.preprocess_image(img)
                processed_images.append(proc_img)
            
            print("✅ 預處理完成，圖片尺寸:", processed_images[0].size)

            # 2. 執行生成
            print("🧊 [Step 2] 開始 3D 生成...")
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
            
            # --- 產出檔案 ---
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

            # 清理暫存的 CUDA 記憶體 (不是卸載模型，只是整理碎片)
            torch.cuda.empty_cache()

            return {
                "gaussian_video": f"/download/{request_id}/gs.mp4",
                "radiance_video": f"/download/{request_id}/rf.mp4",
                "mesh_video": f"/download/{request_id}/mesh.mp4",
                "glb_file": f"/download/{request_id}/output.glb",
                "ply_file": f"/download/{request_id}/output.ply"
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            raise e
        
        # [選項] 如果你想在每次推理完後完全釋放模型以節省 VRAM，
        # 請取消註解以下 finally 區塊 (這會導致下一次請求變慢)：
        finally:
            # global pipeline
            # del pipeline
            # pipeline = None
            # gc.collect()
            torch.cuda.empty_cache()
            # print("🧹 模型已卸載")

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
