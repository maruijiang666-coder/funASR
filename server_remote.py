#!/usr/bin/env python3
"""
实时语音识别 + 说话人分离 Web 服务

启动时四模型常驻 cuda:0:
  - Fun-ASR-Nano-2512   (录音模式 / 短问句, 流式, 自带 VAD)
  - Paraformer-online   (会议模式 / 长会议, 流式)
  - FSMN-VAD            (会议模式 VAD 切段 + 静音检测)
  - cam++ spk           (会议模式说话人 embedding)

前端统一推 PCM Float32 (16kHz mono) 至 WebSocket;
服务端按 mode 路由到不同 ASR 引擎, 段结果 + 说话人 → 实时推回前端。

会议模式额外把每段写入 meetings/{mid}.json。
"""
import os
import sys
import json
import asyncio
import time
import uuid
from datetime import datetime
from collections import deque

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import re
from pathlib import Path
import httpx
from sklearn.cluster import AgglomerativeClustering

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "1")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # /home/quant/data/dev/funasr-venv
MODEL_DIR = os.path.join(BASE, "model")
CACHE_DIR = os.path.join(MODEL_DIR, ".cache")
INDEX_HTML = os.path.join(BASE, "web", "index.html")
MEETINGS_DIR = os.path.join(BASE, "web", "meetings")
RECORDINGS_DIR = os.path.join(BASE, "web", "recordings")

os.makedirs(MEETINGS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ===== 本地模型路径 =====
NANO_PATH = os.path.join(MODEL_DIR, "Fun-ASR-Nano-2512")
PARAFORMER_ONLINE_PATH = os.path.join(MODEL_DIR, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
PARAFORMER_OFFLINE_PATH = os.path.join(MODEL_DIR, "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
SENSEVOICE_PATH = os.path.join(MODEL_DIR, "SenseVoiceSmall")
VAD_PATH = os.path.join(CACHE_DIR, "models",
                        "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        "snapshots", "master")
SPK_PATH = os.path.join(CACHE_DIR, "models",
                        "iic--speech_campplus_sv_zh-cn_16k-common",
                        "snapshots", "master")

SAMPLE_RATE = 16000

# ===== 全局模型 (主进程初始化) =====
nano_asr = None            # Fun-ASR-Nano (vllm 后端, 录音模式 + 文件转写共用)
para_asr = None            # Paraformer ONLINE (会议模式流式, 不可用于文件)
para_offline_asr = None    # Paraformer VAD-PUNC (文件转写, 自带 VAD 切句 + 标点)
sensevoice_asr = None      # SenseVoice-Small (文件转写 多语种)
vad_model = None           # FSMN-VAD
spk_model = None           # cam++

# 模型就绪状态 (供前端 GET /api/models/status 查询)
MODEL_STATUS = {
    "ready": False,
    "loading": True,
    "models": {
        "nano":      {"loaded": False, "path": NANO_PATH},
        "paraformer":{"loaded": False, "path": PARAFORMER_ONLINE_PATH},
        "vad":       {"loaded": False, "path": VAD_PATH},
        "spk":       {"loaded": False, "path": SPK_PATH},
    },
    "gpu": None,
}


# ===== 工具函数 =====

def _has_speech(audio: np.ndarray, rms_threshold: float = 0.002) -> bool:
    """简单能量检测: 静音段 (RMS 极低) 直接跳过 ASR 调用 (防 funasr-Nano 短静音崩溃)"""
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return rms > rms_threshold


def normalize_pf_sentence(s: str) -> str:
    """去掉 Paraformer 字符级输出里的多余空格（保留标点前的空格）"""
    import re
    if not s:
        return ""
    s = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", s)
    return s.strip()


def extract_sv_embedding(audio: np.ndarray):
    """抽 192 维声纹 embedding (L2 normalized)"""
    try:
        r = spk_model.generate(input=audio, batch_size_s=1, disable_pbar=True)
        if r and isinstance(r, list):
            emb = r[0].get("spk_embedding") if isinstance(r[0], dict) else None
            if emb is None:
                return None
            e = emb
            if hasattr(e, "detach"):
                e = e.detach().cpu().numpy()
            e = np.asarray(e)
            if e.ndim == 2:
                e = e.mean(axis=0)
            n = np.linalg.norm(e) + 1e-8
            return (e / n).astype(np.float32)
    except Exception as ex:
        print(f"[spk] err: {ex}", flush=True)
    return None


def cluster_speakers(emb_list):
    """滑动窗口聚类"""
    n = len(emb_list)
    if n == 0: return []
    if n == 1: return [0]
    X = np.stack(emb_list, axis=0)
    try:
        clu = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.5,
            metric="cosine", linkage="average",
        )
        labels = clu.fit_predict(X)
    except Exception:
        labels = np.zeros(n, dtype=int)
    return labels.tolist()


def _take_chunk(buffer: list, n_samples: int):
    """从 buffer 头部取 n_samples 个样本, 原地修改 buffer"""
    taken = []
    remaining = n_samples
    while remaining > 0 and buffer:
        head = buffer[0]
        if head.size <= remaining:
            taken.append(head)
            remaining -= head.size
            buffer.pop(0)
        else:
            taken.append(head[:remaining])
            buffer[0] = head[remaining:]
            remaining = 0
    return np.concatenate(taken) if taken else np.zeros(0, dtype=np.float32)


def get_gpu_info():
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3, text=True,
        ).stdout.strip()
        if out:
            name, used, total, util = [x.strip() for x in out.split(",")]
            return {"name": name, "used_mb": int(used), "total_mb": int(total), "util_pct": int(util)}
    except Exception:
        pass
    return None


# ===== 模型初始化 =====

def init_models():
    """主进程加载所有模型 (顺序: 先轻量后重量, 让 vllm 占大块 GPU 最后)"""
    global nano_asr, para_asr, para_offline_asr, sensevoice_asr, vad_model, spk_model
    from funasr import AutoModel

    print("[init] VAD (fsmn-vad) ...", flush=True)
    vad_model = AutoModel(model=VAD_PATH, disable_update=True, disable_log=True, device="cuda:0")
    MODEL_STATUS["models"]["vad"]["loaded"] = True

    print("[init] speaker embedding (cam++) ...", flush=True)
    spk_model = AutoModel(model=SPK_PATH, disable_update=True, disable_log=True, device="cuda:0")
    MODEL_STATUS["models"]["spk"]["loaded"] = True

    print("[init] Paraformer-online (会议模式流式) ...", flush=True)
    # 不配置 vad_model: 流式识别不需要服务端 VAD (chunk 自己切段)
    # spk_model 也独立加载, 不用 inference_with_vad 路径 (那段 chunk_size 语义冲突)
    para_asr = AutoModel(
        model=PARAFORMER_ONLINE_PATH,
        device="cuda:0",
        disable_update=True,
        disable_log=True,
    )
    MODEL_STATUS["models"]["paraformer"]["loaded"] = True

    print("[init] Paraformer-vad-punc (文件转写 离线) ...", flush=True)
    # 文件场景: 离线 -vad-punc 版 (带 VAD + 标点 + 时间戳), 与 online 版并存
    para_offline_asr = AutoModel(
        model=PARAFORMER_OFFLINE_PATH,
        device="cuda:0",
        vad_model=VAD_PATH,
        disable_update=True,
        disable_log=True,
    )
    MODEL_STATUS["models"]["paraformer_offline"] = {"loaded": True, "path": PARAFORMER_OFFLINE_PATH}

    print("[init] SenseVoice-Small (文件转写 多语种) ...", flush=True)
    sensevoice_asr = AutoModel(
        model=SENSEVOICE_PATH,
        device="cuda:0",
        disable_update=True,
        disable_log=True,
    )
    MODEL_STATUS["models"]["sensevoice"] = {"loaded": True, "path": SENSEVOICE_PATH}

    print("[init] Fun-ASR-Nano-2512 (vLLM 后端) ...", flush=True)
    # 注意: FunASRNano 通过 AutoModel 加载后 inference 路径有 bug (data_in[0] 变成 None)
    # 必须用 inference_vllm.FunASRNanoVLLM.from_pretrained 走 vLLM 后端
    # 关键: 挂 vad_model=VAD_PATH 走 inference_with_vad 路径, 不挂则裸跑 LLM 续写极易出幻觉循环
    from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
    nano_asr = FunASRNanoVLLM.from_pretrained(
        model=NANO_PATH,
        device="cuda:0",
        dtype="bf16",
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        vad_model=VAD_PATH,
    )
    MODEL_STATUS["models"]["nano"]["loaded"] = True

    MODEL_STATUS["ready"] = True
    MODEL_STATUS["loading"] = False
    MODEL_STATUS["gpu"] = get_gpu_info()
    print(f"[init] 全部就绪 · GPU: {MODEL_STATUS['gpu']}", flush=True)


# ===== 启动时清理 vLLM EngineCore 孤儿 =====
def kill_vllm_orphans():
    """vLLM 主进程被 kill 时, EngineCore 子进程会留在 GPU 上不退出.
    下次启动时 vLLM 加载报 Free memory < desired utilization. 这里预清."""
    import subprocess
    try:
        out = subprocess.run(
            ["pgrep", "-f", "VLLM::EngineCore"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if out:
            print(f"[init] 发现 {len(out.splitlines())} 个 vLLM EngineCore 孤儿进程, 清理中...", flush=True)
            subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], capture_output=True, timeout=3)
            import time as _t; _t.sleep(3)
        out2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        used = int(out2.splitlines()[0]) if out2 else -1
        if used > 200:
            print(f"[init] GPU 仍占用 {used} MiB, 等 5s 释放...", flush=True)
            import time as _t; _t.sleep(5)
    except Exception as e:
        print(f"[init] orphan check 跳过: {e}", flush=True)


# ===== 启动时清理 vLLM EngineCore 孤儿 =====



def make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index():
        return FileResponse(INDEX_HTML)

    @app.get("/health")
    async def health():
        return {"status": "ok", "ts": time.time()}

    @app.get("/api/models/status")
    async def models_status():
        MODEL_STATUS["gpu"] = get_gpu_info()
        return MODEL_STATUS

    # ---- 文件转写 (上传 / URL) ----

    import threading
    import queue as _q
    import uuid as _uuid
    import httpx

    TRANSCRIPTS_DIR = os.path.join(BASE, "web", "transcripts")
    UPLOADS_DIR = os.path.join(BASE, "web", "uploads")
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # 任务池: job_id -> {status, events: queue, result}
    JOBS: dict = {}
    JOBS_LOCK = threading.Lock()

    # 走 BackgroundTasks + 独立 thread 跑 worker (worker 内 funasr 是阻塞的)
    import sys as _sys
    _WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
    if _WORKER_DIR not in _sys.path:
        _sys.path.insert(0, _WORKER_DIR)

    def _job_progress_cb(job_id: str):
        def cb(stage: str, pct: int, msg: str):
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["events"].put({"stage": stage, "pct": pct, "msg": msg})
        return cb

    def _job_thread(job_id: str, input_path: str, output_prefix: str, model: str):
        try:
            import transcribe_worker
            # 复用 server 启动时已加载的模型, 不再 from_pretrained (避免再启 vLLM 锁 GPU)
            if model == "paraformer":
                mi = para_offline_asr
            elif model == "sensevoice":
                mi = sensevoice_asr
            elif model == "nano":
                mi = nano_asr
            else:
                raise ValueError(f"未知模型: {model}")
            res = transcribe_worker.run(
                input_path, output_prefix, model=model, model_instance=mi,
                progress_cb=_job_progress_cb(job_id),
            )
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["status"] = "done"
                    JOBS[job_id]["result"] = res
                    JOBS[job_id]["events"].put({"stage": "done", "pct": 100, "msg": "完成", "result": res})
        except Exception as e:
            import traceback
            err = f"{e}\n{traceback.format_exc()}"
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["error"] = err
                    JOBS[job_id]["events"].put({"stage": "error", "pct": -1, "msg": str(e)})

    @app.post("/api/transcribe/upload")
    async def transcribe_upload(file: UploadFile = File(...), model: str = Form("paraformer")):
        """上传文件 → 启动后台转写 → 返回 job_id. 进度走 /api/transcribe/{job_id}/events"""
        if model not in ("paraformer", "sensevoice", "nano"):
            return JSONResponse({"error": f"未知模型: {model}"}, status_code=400)
        # 保存上传文件
        job_id = _uuid.uuid4().hex[:12]
        suffix = Path(file.filename or "").suffix or ".bin"
        safe_name = re.sub(r"[^\w.\-]", "_", file.filename or "upload")[:80] or "upload"
        save_path = os.path.join(UPLOADS_DIR, f"{job_id}_{safe_name}")
        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

        # 输出前缀: transcripts/<job_id>_<base>
        base = Path(safe_name).stem
        output_prefix = os.path.join(TRANSCRIPTS_DIR, f"{job_id}_{base}")

        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "running",
                "events": _q.Queue(maxsize=1024),
                "result": None,
                "error": None,
                "input": file.filename,
                "model": model,
                "created_at": time.time(),
            }
        threading.Thread(target=_job_thread, args=(job_id, save_path, output_prefix, model), daemon=True).start()
        return {"job_id": job_id, "input": file.filename, "model": model}

    @app.post("/api/transcribe/url")
    async def transcribe_url(req: Request):
        """URL → 下载 → 启动转写 → 返回 job_id"""
        try:
            body = await req.json()
        except Exception:
            return JSONResponse({"error": "bad json"}, status_code=400)
        url = (body.get("url") or "").strip()
        model = (body.get("model") or "paraformer").lower()
        if not url:
            return JSONResponse({"error": "缺少 url"}, status_code=400)
        if model not in ("paraformer", "sensevoice", "nano"):
            return JSONResponse({"error": f"未知模型: {model}"}, status_code=400)

        job_id = _uuid.uuid4().hex[:12]
        # 从 URL 推断文件名
        url_name = url.split("?")[0].rstrip("/").split("/")[-1] or "remote"
        safe_name = re.sub(r"[^\w.\-]", "_", url_name)[:80]
        save_path = os.path.join(UPLOADS_DIR, f"{job_id}_{safe_name}")
        output_prefix = os.path.join(TRANSCRIPTS_DIR, f"{job_id}_{Path(safe_name).stem}")

        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "running",
                "events": _q.Queue(maxsize=1024),
                "result": None,
                "error": None,
                "input": url,
                "model": model,
                "created_at": time.time(),
            }

        def _download_and_run():
            try:
                # 进度: 下载
                with JOBS_LOCK:
                    JOBS[job_id]["events"].put({"stage": "downloading", "pct": 2, "msg": f"下载 {url}"})
                with httpx.Client(timeout=180, follow_redirects=True) as cli:
                    with cli.stream("GET", url) as r:
                        r.raise_for_status()
                        total = int(r.headers.get("content-length", 0))
                        got = 0
                        with open(save_path, "wb") as f:
                            for chunk in r.iter_bytes(chunk_size=1 << 20):
                                f.write(chunk)
                                got += len(chunk)
                                if total:
                                    pct = 2 + int(5 * got / total)
                                    with JOBS_LOCK:
                                        JOBS[job_id]["events"].put({
                                            "stage": "downloading", "pct": pct,
                                            "msg": f"下载 {got//1024}/{total//1024} KB"
                                        })
                # 启动识别
                _job_thread(job_id, save_path, output_prefix, model)
            except Exception as e:
                import traceback
                err = f"{e}\n{traceback.format_exc()}"
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id]["status"] = "error"
                        JOBS[job_id]["error"] = err
                        JOBS[job_id]["events"].put({"stage": "error", "pct": -1, "msg": str(e)})

        threading.Thread(target=_download_and_run, daemon=True).start()
        return {"job_id": job_id, "input": url, "model": model}

    @app.get("/api/transcribe/{job_id}/events")
    async def transcribe_events(job_id: str, req: Request):
        """SSE 推流: 实时进度"""
        async def gen():
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'msg':'job not found'}, ensure_ascii=False)}\n\n"
                return
            # 推历史 (没有, 直接推流)
            while True:
                if await req.is_disconnected():
                    return
                try:
                    ev = job["events"].get(timeout=15)
                except _q.Empty:
                    # 心跳
                    yield ": heartbeat\n\n"
                    continue
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev.get("stage") in ("done", "error"):
                    return
        from fastapi.responses import StreamingResponse
        return StreamingResponse(gen(), media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/transcribe/{job_id}/result")
    async def transcribe_result(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            return JSONResponse({"error": "job not found"}, status_code=404)
        if job["status"] != "done":
            return JSONResponse({"error": "not done", "status": job["status"]}, status_code=400)
        return job["result"]

    @app.get("/api/transcribe/{job_id}/download/{fmt}")
    async def transcribe_download(job_id: str, fmt: str):
        """fmt = 'srt' | 'json'  下载文件名带模型后缀, 例: xxx_paraformer.srt"""
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job or not job.get("result"):
            return JSONResponse({"error": "not done"}, status_code=400)
        path = job["result"].get(fmt)
        if not path or not os.path.exists(path):
            return JSONResponse({"error": f"{fmt} not found"}, status_code=404)
        # 物理文件名 = <job_id>_<base>.<ext>
        # 下载文件名 = <base>_<model>.<ext>  (带模型, 用户一眼能区分)
        ext = "." + fmt
        base = os.path.basename(path)[: -len(ext)]
        # 去掉 job_id 前缀 (12 字符 + 下划线)
        if "_" in base and len(base) > 13 and base[12] == "_":
            base = base[13:]
        model = job.get("model", "unknown")
        download_name = f"{base}_{model}{ext}"
        return FileResponse(
            path, filename=download_name,
            media_type="application/octet-stream",
        )

    # ---- 会议 JSON CRUD ----

    @app.post("/api/meeting/new")
    async def meeting_new(req: Request):
        try:
            body = await req.json() if (await req.body()) else {}
        except Exception:
            body = {}
        mid = uuid.uuid4().hex[:12]
        meta = {
            "id": mid,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "title": body.get("title", f"会议-{mid[:6]}"),
            "speakers": {},
            "segments": [],
        }
        path = os.path.join(MEETINGS_DIR, f"{mid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return {"meeting_id": mid, "path": path}

    @app.post("/api/meeting/{mid}/append")
    async def meeting_append(mid: str, req: Request):
        path = os.path.join(MEETINGS_DIR, f"{mid}.json")
        if not os.path.exists(path):
            return JSONResponse({"error": "meeting not found"}, status_code=404)
        try:
            body = await req.json()
        except Exception:
            return JSONResponse({"error": "bad json"}, status_code=400)
        seg_id = body.get("seg")
        with open(path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            # 按 seg 编号去重: 已存在则更新 (合并 spk_name), 不存在则追加
            existing_idx = None
            for i, s in enumerate(data["segments"]):
                if s.get("seg") == seg_id:
                    existing_idx = i
                    break
            if existing_idx is not None:
                # 合并: 保留已有字段, 用新字段覆盖 (主要是更新 spk_name)
                for k, v in body.items():
                    if v is not None:
                        data["segments"][existing_idx][k] = v
            else:
                data["segments"].append(body)
            if body.get("spk_name"):
                data["speakers"][str(body["spk"])] = body["spk_name"]
            f.seek(0); f.truncate()
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"ok": True, "count": len(data["segments"])}

    @app.post("/api/meeting/{mid}/speaker")
    async def meeting_speaker(mid: str, req: Request):
        path = os.path.join(MEETINGS_DIR, f"{mid}.json")
        if not os.path.exists(path):
            return JSONResponse({"error": "meeting not found"}, status_code=404)
        body = await req.json()
        with open(path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["speakers"][str(body["spk"])] = body["name"]
            f.seek(0); f.truncate()
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"ok": True}

    @app.post("/api/meeting/{mid}/finish")
    async def meeting_finish(mid: str):
        path = os.path.join(MEETINGS_DIR, f"{mid}.json")
        if not os.path.exists(path):
            return JSONResponse({"error": "meeting not found"}, status_code=404)
        with open(path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["finished_at"] = datetime.now().isoformat(timespec="seconds")
            f.seek(0); f.truncate()
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    @app.get("/api/meeting/{mid}/download")
    async def meeting_download(mid: str):
        path = os.path.join(MEETINGS_DIR, f"{mid}.json")
        if not os.path.exists(path):
            return JSONResponse({"error": "meeting not found"}, status_code=404)
        return FileResponse(
            path, media_type="application/json",
            filename=f"meeting_{mid}.json",
        )

    @app.delete("/api/meeting/{mid}")
    async def meeting_delete(mid: str):
        path = os.path.join(MEETINGS_DIR, f"{mid}.json")
        if not os.path.exists(path):
            return JSONResponse({"error": "meeting not found"}, status_code=404)
        try:
            os.remove(path)
            return {"ok": True, "deleted": mid}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/meetings")
    async def meetings_list():
        out = []
        for f in sorted(os.listdir(MEETINGS_DIR), reverse=True):
            if not f.endswith(".json"): continue
            try:
                with open(os.path.join(MEETINGS_DIR, f), encoding="utf-8") as fp:
                    d = json.load(fp)
                out.append({
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "created_at": d.get("created_at"),
                    "finished_at": d.get("finished_at"),
                    "segments": len(d.get("segments", [])),
                    "speakers": len(d.get("speakers", {})),
                })
            except Exception:
                pass
        return out

    @app.get("/api/recording/{filename}/download")
    async def recording_download(filename: str):
        """下载录音 WAV 文件"""
        # 防路径穿越
        if "/" in filename or ".." in filename or "\\" in filename or not filename.endswith(".wav"):
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = os.path.join(RECORDINGS_DIR, filename)
        if not os.path.exists(path):
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, filename=filename, media_type="audio/wav")

    @app.get("/api/recordings")
    async def recordings_list():
        """列出所有录音文件"""
        out = []
        for f in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
            if not f.endswith(".wav"):
                continue
            try:
                st = os.stat(os.path.join(RECORDINGS_DIR, f))
                out.append({
                    "filename": f,
                    "size_mb": round(st.st_size / 1024 / 1024, 2),
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                })
            except Exception:
                pass
        return out

    # ---- WebSocket: 录制到 WAV + 流式 ASR + 结束时跑离线 ASR ----
    #
    # 流程:
    #   1. 客户端推 PCM Float32 binary → 服务端写 16k/mono/16bit WAV (永久保存到 web/recordings/)
    #   2. 后台 task 实时流式 ASR (para_asr, 每 600ms 一个 chunk), 推 streaming_text 给前端
    #      → 用户录制中就能实时看到识别结果, caption-bar 实时更新
    #   3. 客户端发 stop → 关 WAV + 取消后台 task
    #   4. 跑最终 ASR (transcribe_worker, 整文件) 拿精确段 (带标点)
    #   5. 会议模式写 meeting JSON (含 recording_path)
    #   6. 推 asr_done (含 recording_filename) → 关闭 WS
    #
    # 录音文件不删, 客户端可通过 /api/recording/{filename}/download 下载.

    import wave as _wave

    async def _run_final_asr(wav_path: str, mode: str, meeting_id, ws: WebSocket, seg_count_ref: list):
        """跑最终 ASR (transcribe_worker, 整文件), 推剩余 segment, 会议模式写 meeting JSON.
        返回 (新增段数, asr_dur)."""
        import transcribe_worker

        loop = asyncio.get_event_loop()
        output_prefix = wav_path[:-4] + "_out"

        def progress_cb(stage: str, pct: int, msg: str):
            try:
                asyncio.run_coroutine_threadsafe(
                    ws.send_json({"type": "asr_progress", "stage": stage, "pct": pct, "msg": msg}),
                    loop,
                )
            except Exception:
                pass

        result = await loop.run_in_executor(
            None,
            lambda: transcribe_worker.run(
                wav_path, output_prefix,
                model="paraformer", model_instance=para_offline_asr,
                progress_cb=progress_cb,
            ),
        )

        new_segments = []
        json_path = result.get("json")
        if json_path and os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            sents = (data[0].get("sentence_info") or []) if data else []
            for s in sents:
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                start_ms = int(s.get("start", 0))
                end_ms = int(s.get("end", 0))
                seg_count_ref[0] += 1
                seg = {
                    "type": "segment",
                    "seg": seg_count_ref[0],
                    "text": text,
                    "audio_len": round((end_ms - start_ms) / 1000, 2),
                    "asr_dur": 0,
                    "spk": 0,
                    "spk_name": None,
                    "engine": "paraformer-vad-punc",
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
                new_segments.append(seg)

            # ---- CAM++ 说话人分离 ----
            if len(new_segments) >= 2:
                try:
                    import soundfile as sf
                    full_audio, _ = sf.read(wav_path, dtype="float32")
                    emb_list = []
                    valid_indices = []
                    for i, seg in enumerate(new_segments):
                        s = int(seg["start_ms"] * SAMPLE_RATE / 1000)
                        e = int(seg["end_ms"] * SAMPLE_RATE / 1000)
                        chunk = full_audio[s:e]
                        if chunk.size < SAMPLE_RATE * 0.3:
                            emb_list.append(None)
                            continue
                        emb = await loop.run_in_executor(
                            None, lambda c=chunk: extract_sv_embedding(c),
                        )
                        emb_list.append(emb)
                        valid_indices.append(i)
                    valid_embs = [emb_list[i] for i in valid_indices]
                    if len(valid_embs) >= 2:
                        labels = cluster_speakers(valid_embs)
                        for idx, label in zip(valid_indices, labels):
                            new_segments[idx]["spk"] = int(label)
                    elif len(valid_embs) == 1:
                        new_segments[valid_indices[0]]["spk"] = 0
                    print(f"[diarization] {len(valid_embs)} segs -> {len(set(s['spk'] for s in new_segments))} speakers", flush=True)
                except Exception as ex:
                    print(f"[diarization] err (fallback spk=0): {ex}", flush=True)

            # 发送给前端
            for seg in new_segments:
                await ws.send_json(seg)

        # 会议模式: 整段写 meeting JSON (含 recording_path)
        if mode == "meeting" and meeting_id:
            meeting_path = os.path.join(MEETINGS_DIR, f"{meeting_id}.json")
            if os.path.exists(meeting_path):
                with open(meeting_path, "r+", encoding="utf-8") as f:
                    mdata = json.load(f)
                    mdata["segments"] = [
                        {
                            "seg": seg["seg"],
                            "text": seg["text"],
                            "start_ms": seg["start_ms"],
                            "end_ms": seg["end_ms"],
                            "audio_len": seg["audio_len"],
                            "engine": seg["engine"],
                            "spk": seg["spk"],
                            "spk_name": seg.get("spk_name"),
                        }
                        for seg in new_segments
                    ]
                    f.seek(0); f.truncate()
                    json.dump(mdata, f, ensure_ascii=False, indent=2)

        return len(new_segments), result.get("asr_dur", 0)

    @app.websocket("/ws/asr")
    async def ws_asr(ws: WebSocket):
        await ws.accept()

        # 首条控制消息: {"mode": "record" | "meeting", "meeting_id": "..."}
        try:
            first = await ws.receive_text()
            ctrl = json.loads(first)
        except Exception:
            await ws.close(code=4000)
            return

        mode = ctrl.get("mode", "record")
        meeting_id = ctrl.get("meeting_id")

        # 永久保存到 web/recordings/, 会议模式复用 meeting_id, 录音模式用时间戳+随机
        if mode == "meeting" and meeting_id:
            audio_filename = f"meeting_{meeting_id}.wav"
        else:
            audio_filename = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"
        wav_path = os.path.join(RECORDINGS_DIR, audio_filename)

        await ws.send_json({
            "type": "ready",
            "mode": mode,
            "engine": "Paraformer-Streaming",
        })
        print(f"[ws] record start: mode={mode} meeting={meeting_id} file={audio_filename}", flush=True)

        loop = asyncio.get_event_loop()
        stop_flag = {"v": False}
        asr_lock = asyncio.Lock()

        # 打开 WAV 文件
        wav_file = _wave.open(wav_path, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        # 流式 ASR 状态
        stream_cache = {}          # Paraformer streaming cache (模型内部管理 prev_samples)
        stream_sent_text = ""      # 已推给前端的文本 (用于 diff)
        new_pcm_buffer: list = []  # 自上次流式 ASR 以来的新 PCM 块
        new_pcm_total = 0          # buffer 中总样本数
        total = 0                  # 录音总样本数
        seg_count_ref = [0]
        last_heartbeat = 0.0

        # 流式 ASR 参数 (600ms 一个 chunk)
        STREAM_CHUNK_SIZE = [0, 10, 5]
        STREAM_CHUNK_STRIDE = STREAM_CHUNK_SIZE[1] * 960  # 9600 samples = 600ms
        STREAM_ENCODER_LOOK_BACK = 4
        STREAM_DECODER_LOOK_BACK = 1

        async def streaming_asr_task():
            """实时流式 ASR: 每 ~200ms 检查 buffer, 有新音频就喂给 para_asr.
            模型返回每个 chunk 的文本片段 (非累积), 需要手动拼接."""
            nonlocal new_pcm_total, stream_sent_text
            while not stop_flag["v"]:
                await asyncio.sleep(0.2)
                if stop_flag["v"]:
                    break
                if not new_pcm_buffer or new_pcm_total == 0:
                    continue
                audio = np.concatenate(new_pcm_buffer)
                new_pcm_buffer.clear()
                new_pcm_total = 0
                try:
                    res = await loop.run_in_executor(
                        None,
                        lambda a=audio: para_asr.generate(
                            input=a,
                            cache=stream_cache,
                            is_final=False,
                            chunk_size=STREAM_CHUNK_SIZE,
                            encoder_chunk_look_back=STREAM_ENCODER_LOOK_BACK,
                            decoder_chunk_look_back=STREAM_DECODER_LOOK_BACK,
                        ),
                    )
                except Exception as e:
                    print(f"[streaming ASR] err: {e}", flush=True)
                    continue
                if not (res and isinstance(res, list) and res and isinstance(res[0], dict)):
                    continue
                chunk_text = normalize_pf_sentence(res[0].get("text", ""))
                if not chunk_text:
                    continue
                # 模型返回当前 chunk 的文本片段, 直接拼接到累积文本
                stream_sent_text += chunk_text
                await ws.send_json({
                    "type": "streaming_text",
                    "text": stream_sent_text,
                    "new_text": chunk_text,
                })

        # 启动流式 ASR 后台 task
        streaming_task = asyncio.create_task(streaming_asr_task())

        try:
            while not stop_flag["v"]:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"]:
                    pcm = np.frombuffer(msg["bytes"], dtype=np.float32)
                    if pcm.size == 0:
                        continue
                    pcm_int16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
                    wav_file.writeframes(pcm_int16.tobytes())
                    new_pcm_buffer.append(pcm.copy())
                    new_pcm_total += pcm.size
                    total += pcm.size

                    # 每 2s 推一次心跳给前端显示录制时长
                    now = time.time()
                    if now - last_heartbeat > 2:
                        last_heartbeat = now
                        await ws.send_json({
                            "type": "recording",
                            "duration_s": round(total / SAMPLE_RATE, 1),
                        })
                elif "text" in msg and msg["text"]:
                    try:
                        d = json.loads(msg["text"])
                        if d.get("type") == "stop":
                            stop_flag["v"] = True
                            break
                    except Exception:
                        pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ws record loop] err: {e}", flush=True)
        finally:
            stop_flag["v"] = True
            try:
                await asyncio.wait_for(streaming_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                streaming_task.cancel()
            try: wav_file.close()
            except: pass

        duration = total / SAMPLE_RATE
        print(f"[ws] record end: {duration:.1f}s file={audio_filename}", flush=True)

        if total < SAMPLE_RATE * 0.5:
            await ws.send_json({"type": "asr_error", "error": "录音太短 (< 0.5s)"})
            try: await ws.close()
            except: pass
            return

        # 流式 ASR 最终 flush (is_final=True)
        if new_pcm_buffer and new_pcm_total > 0:
            audio = np.concatenate(new_pcm_buffer)
            try:
                res = await loop.run_in_executor(
                    None,
                    lambda a=audio: para_asr.generate(
                        input=a,
                        cache=stream_cache,
                        is_final=True,
                        chunk_size=STREAM_CHUNK_SIZE,
                        encoder_chunk_look_back=STREAM_ENCODER_LOOK_BACK,
                        decoder_chunk_look_back=STREAM_DECODER_LOOK_BACK,
                    ),
                )
                if res and isinstance(res, list) and res and isinstance(res[0], dict):
                    chunk_text = normalize_pf_sentence(res[0].get("text", ""))
                    if chunk_text:
                        stream_sent_text += chunk_text
                        await ws.send_json({
                            "type": "streaming_text",
                            "text": stream_sent_text,
                            "new_text": chunk_text,
                        })
            except Exception as e:
                print(f"[streaming ASR final flush] err: {e}", flush=True)

        # 跑最终 ASR (整文件, 带标点的精确版本)
        await ws.send_json({"type": "asr_started", "duration_s": round(duration, 1)})
        try:
            async with asr_lock:
                new_count, asr_dur = await _run_final_asr(wav_path, mode, meeting_id, ws, seg_count_ref)
            print(f"[ws] final ASR done: +{new_count} segs, total={seg_count_ref[0]}", flush=True)
            await ws.send_json({
                "type": "asr_done",
                "count": seg_count_ref[0],
                "audio_dur": round(duration, 1),
                "asr_dur": round(asr_dur, 1),
                "recording_filename": audio_filename,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            await ws.send_json({"type": "asr_error", "error": str(e)})

        try: await ws.close()
        except: pass

    return app


if __name__ == "__main__":
    import ssl
    kill_vllm_orphans()  # 清上一次 vLLM 残留的 EngineCore 孤儿进程
    init_models()
    app = make_app()

    CERT_DIR = os.path.join(BASE, "web", "certs")
    cert_file = os.path.join(CERT_DIR, "cert.pem")
    key_file = os.path.join(CERT_DIR, "key.pem")
    use_ssl = os.path.exists(cert_file) and os.path.exists(key_file)

    if use_ssl:
        print("[init] uvicorn HTTPS :7443 (also HTTP :7777 -> redirect)", flush=True)
        import threading
        def http_redirect():
            redirect_app = FastAPI()
            @redirect_app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE","PATCH","HEAD","OPTIONS","WS","WSS"])
            async def redir(path: str = ""):
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url=f"https://{os.environ.get('LAN_HOST','6.6.6.66')}:7443/{path}", status_code=308)
            uvicorn.run(redirect_app, host="0.0.0.0", port=7777, log_level="error")
        threading.Thread(target=http_redirect, daemon=True).start()
        uvicorn.run(
            app, host="0.0.0.0", port=7443,
            ssl_certfile=cert_file, ssl_keyfile=key_file,
            log_level="warning",
        )
    else:
        print("[init] uvicorn HTTP :7777 (no cert)", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=7777, log_level="warning")