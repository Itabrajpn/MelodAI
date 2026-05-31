
import os, uuid, time, threading, subprocess, shutil, mimetypes
from pathlib import Path
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory

load_dotenv()

APP_NAME = "MelodAI PRO MultiEngine FIX"
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

PYTHON_EXE = os.getenv("PYTHON_EXE", "python").strip()
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", "").strip()

WAV2LIP_DIR = Path(os.getenv("WAV2LIP_DIR", BASE_DIR / "Wav2Lip")).resolve()
SADTALKER_DIR = Path(os.getenv("SADTALKER_DIR", BASE_DIR / "SadTalker")).resolve()
MUSETALK_DIR = Path(os.getenv("MUSETALK_DIR", BASE_DIR / "MuseTalk")).resolve()
MUSETALK_SCRIPT = os.getenv("MUSETALK_SCRIPT", "").strip()

jobs = {}

def allowed_file(filename, exts):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in exts

def update(job_id, **kw):
    jobs[job_id].update(kw)

def run_cmd(job_id, cmd, cwd=None):
    update(job_id, message="Executando motor local...", progress=20)
    p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, universal_newlines=True)
    lines = []
    for line in p.stdout:
        line = line.rstrip()
        if line:
            lines.append(line[-700:])
            jobs[job_id]["log"] = "\n".join(lines[-50:])
    p.wait()
    return p.returncode == 0

def ffmpeg_convert(src, dst, args):
    cmd = ["ffmpeg", "-y", "-i", str(src)] + args + [str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and dst.exists()

# ---------------- HEYGEN V3 ----------------
def hx(json=False):
    h = {"x-api-key": HEYGEN_API_KEY}
    if json:
        h["Content-Type"] = "application/json"
    return h

def heygen_upload(path):
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    with open(path, "rb") as f:
        r = requests.post(
            "https://api.heygen.com/v3/assets",
            headers={"x-api-key": HEYGEN_API_KEY},
            files={"file": (path.name, f, mime)},
            timeout=300,
        )
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Upload HeyGen HTTP {r.status_code}: {j}")
    data = j.get("data") or {}
    asset_id = data.get("asset_id")
    if not asset_id:
        raise RuntimeError(f"HeyGen upload sem asset_id: {j}")
    return asset_id

def run_heygen(job_id, face_path, audio_path, output_path, opt):
    if not HEYGEN_API_KEY:
        raise RuntimeError("HEYGEN_API_KEY não configurada no arquivo .env")

    if face_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        raise RuntimeError("HeyGen Image-to-Video aceita imagem JPG/PNG. Para vídeo, use MuseTalk ou Wav2Lip.")

    update(job_id, engine="HeyGen API", message="Upload imagem HeyGen...", progress=10)
    image_asset_id = heygen_upload(face_path)

    update(job_id, message="Upload áudio HeyGen...", progress=25)
    if audio_path.suffix.lower() not in [".mp3", ".wav"]:
        mp3 = UPLOAD_DIR / f"{job_id}_audio.mp3"
        if ffmpeg_convert(audio_path, mp3, ["-q:a", "0"]):
            audio_path = mp3
    audio_asset_id = heygen_upload(audio_path)

    resolution = opt.get("resolution", "720p")
    aspect = opt.get("aspect_ratio", "16:9")
    background_color = opt.get("background_color") or "#000000"

    payload = {
        "type": "image",
        "image": {"type": "asset_id", "asset_id": image_asset_id},
        "audio_asset_id": audio_asset_id,
        "title": opt.get("title") or f"MelodAI {job_id}",
        "resolution": resolution,
        "aspect_ratio": aspect,
        "fit": opt.get("fit", "contain"),
        "background": {"value": background_color},
        "output_format": "mp4"
    }

    if opt.get("remove_background"):
        payload["remove_background"] = True

    update(job_id, message="Criando vídeo HeyGen...", progress=40)
    r = requests.post("https://api.heygen.com/v3/videos", headers=hx(json=True), json=payload, timeout=180)
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}

    if r.status_code not in (200, 201):
        raise RuntimeError(f"HeyGen HTTP {r.status_code}: {j}")

    video_id = (j.get("data") or {}).get("video_id")
    if not video_id:
        raise RuntimeError(f"HeyGen não retornou video_id: {j}")

    update(job_id, message=f"HeyGen processando: {video_id}", progress=50)

    for i in range(180):
        time.sleep(5)
        sr = requests.get(f"https://api.heygen.com/v3/videos/{video_id}", headers=hx(), timeout=90)
        try:
            sj = sr.json()
        except Exception:
            sj = {"raw": sr.text}
        d = sj.get("data") or {}
        status = d.get("status", "")
        msg = d.get("failure_message") or d.get("error") or status
        update(job_id, progress=min(95, 50 + i), message=f"HeyGen: {msg}")
        if status == "completed" and d.get("video_url"):
            vr = requests.get(d["video_url"], stream=True, timeout=300)
            with open(output_path, "wb") as f:
                for chunk in vr.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return
        if status == "failed":
            raise RuntimeError(f"HeyGen falhou: {msg}")
    raise RuntimeError("Timeout HeyGen.")

# ---------------- LOCAL ENGINES ----------------
def run_wav2lip(job_id, face_path, audio_path, output_path, opt):
    infer = WAV2LIP_DIR / "inference.py"
    ckpt = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
    if not infer.exists():
        raise RuntimeError(f"Wav2Lip não encontrado: {infer}. Copie a pasta Wav2Lip para esta pasta ou corrija WAV2LIP_DIR no .env.")
    if not ckpt.exists():
        raise RuntimeError(f"Checkpoint não encontrado: {ckpt}")

    wav = UPLOAD_DIR / f"{job_id}_audio.wav"
    if audio_path.suffix.lower() != ".wav":
        if ffmpeg_convert(audio_path, wav, ["-ar", "16000", "-ac", "1"]):
            audio_path = wav

    cmd = [PYTHON_EXE, str(infer), "--checkpoint_path", str(ckpt),
           "--face", str(face_path), "--audio", str(audio_path), "--outfile", str(output_path),
           "--resize_factor", "1", "--pads", "0", "10", "0", "0"]
    ok = run_cmd(job_id, cmd, cwd=WAV2LIP_DIR)
    if not ok or not output_path.exists():
        raise RuntimeError("Wav2Lip falhou. Veja o log.")

def run_sadtalker(job_id, face_path, audio_path, output_path, opt):
    infer = SADTALKER_DIR / "inference.py"
    if not infer.exists():
        raise RuntimeError(f"SadTalker não encontrado: {infer}. Copie a pasta SadTalker para esta pasta ou corrija SADTALKER_DIR no .env.")
    result_dir = OUTPUT_DIR / f"{job_id}_sadtalker"
    result_dir.mkdir(exist_ok=True)

    cmd = [PYTHON_EXE, str(infer), "--driven_audio", str(audio_path),
           "--source_image", str(face_path), "--result_dir", str(result_dir),
           "--still", "--preprocess", "full", "--size", opt.get("sadtalker_size", "256")]
    if opt.get("enhancer"):
        cmd += ["--enhancer", "gfpgan"]

    ok = run_cmd(job_id, cmd, cwd=SADTALKER_DIR)
    mp4s = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ok or not mp4s:
        raise RuntimeError("SadTalker falhou ou não gerou MP4. Veja o log.")
    shutil.copy2(mp4s[0], output_path)

def run_musetalk(job_id, face_path, audio_path, output_path, opt):
    candidates = []
    if MUSETALK_SCRIPT:
        candidates.append(Path(MUSETALK_SCRIPT))
    candidates += [MUSETALK_DIR / "scripts" / "inference.py", MUSETALK_DIR / "inference.py", MUSETALK_DIR / "app.py"]
    script = next((p.resolve() for p in candidates if p.exists()), None)
    if not script:
        raise RuntimeError("MuseTalk não encontrado. Copie a pasta MuseTalk ou configure MUSETALK_SCRIPT no .env.")

    cmd = [PYTHON_EXE, str(script), "--video_path", str(face_path), "--audio_path", str(audio_path), "--result_dir", str(OUTPUT_DIR)]
    ok = run_cmd(job_id, cmd, cwd=MUSETALK_DIR)
    mp4s = sorted([p for p in OUTPUT_DIR.glob("*.mp4") if p.name != output_path.name], key=lambda p: p.stat().st_mtime, reverse=True)
    if not ok or not mp4s:
        raise RuntimeError("MuseTalk falhou ou não gerou MP4. Veja o log.")
    shutil.copy2(mp4s[0], output_path)

def process(job_id, engine, face_path, audio_path, output_path, opt):
    try:
        update(job_id, status="running", progress=5, message=f"Iniciando {engine}...")
        if engine == "heygen": run_heygen(job_id, face_path, audio_path, output_path, opt)
        elif engine == "sadtalker": run_sadtalker(job_id, face_path, audio_path, output_path, opt)
        elif engine == "wav2lip": run_wav2lip(job_id, face_path, audio_path, output_path, opt)
        elif engine == "musetalk": run_musetalk(job_id, face_path, audio_path, output_path, opt)
        else: raise RuntimeError("Motor inválido")
        update(job_id, status="done", progress=100, message="Vídeo gerado com sucesso!", output=output_path.name)
    except Exception as e:
        update(job_id, status="error", progress=100, message=str(e))

@app.route("/")
def index():
    return render_template("index.html", api_configured=bool(HEYGEN_API_KEY))

@app.route("/api/health")
def health():
    return jsonify({
        "heygen_api": bool(HEYGEN_API_KEY),
        "python": PYTHON_EXE,
        "wav2lip_dir": str(WAV2LIP_DIR),
        "wav2lip_found": (WAV2LIP_DIR / "inference.py").exists(),
        "sadtalker_dir": str(SADTALKER_DIR),
        "sadtalker_found": (SADTALKER_DIR / "inference.py").exists(),
        "musetalk_dir": str(MUSETALK_DIR),
        "musetalk_found": any(p.exists() for p in [MUSETALK_DIR / "scripts" / "inference.py", MUSETALK_DIR / "inference.py", MUSETALK_DIR / "app.py"]),
    })

@app.route("/api/generate", methods=["POST"])
def generate():
    if "face" not in request.files or "audio" not in request.files:
        return jsonify({"error": "Envie imagem/vídeo e áudio."}), 400
    face = request.files["face"]
    audio = request.files["audio"]
    engine = request.form.get("engine", "heygen")
    if not allowed_file(face.filename, {"jpg","jpeg","png","mp4","mov","avi"}):
        return jsonify({"error": "Imagem/vídeo: JPG, PNG, MP4, MOV ou AVI."}), 400
    if not allowed_file(audio.filename, {"mp3","wav","aac","m4a"}):
        return jsonify({"error": "Áudio: MP3, WAV, AAC ou M4A."}), 400

    job_id = str(uuid.uuid4())[:10]
    face_path = UPLOAD_DIR / f"{job_id}_face.{face.filename.rsplit('.',1)[1].lower()}"
    audio_path = UPLOAD_DIR / f"{job_id}_audio.{audio.filename.rsplit('.',1)[1].lower()}"
    output_path = OUTPUT_DIR / f"{job_id}_{engine}.mp4"
    face.save(face_path); audio.save(audio_path)

    opt = {
        "title": request.form.get("title", ""),
        "resolution": request.form.get("resolution", "720p"),
        "aspect_ratio": request.form.get("aspect_ratio", "16:9"),
        "fit": request.form.get("fit", "contain"),
        "background_color": request.form.get("background_color", "#000000"),
        "remove_background": request.form.get("remove_background") == "true",
        "enhancer": request.form.get("enhancer") == "true",
        "sadtalker_size": request.form.get("sadtalker_size", "256"),
    }

    jobs[job_id] = {"id":job_id, "engine":engine, "status":"queued", "progress":0,
                    "message":"Na fila...", "output":None, "log":"",
                    "created_at":time.strftime("%Y-%m-%d %H:%M:%S")}
    threading.Thread(target=process, args=(job_id, engine, face_path, audio_path, output_path, opt), daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/api/status/<job_id>")
def status(job_id):
    return jsonify(jobs.get(job_id, {"error":"job not found"}))

@app.route("/api/jobs")
def list_jobs():
    return jsonify(list(jobs.values())[::-1])

@app.route("/outputs/<fn>")
def outputs(fn):
    return send_from_directory(OUTPUT_DIR, fn)

if __name__ == "__main__":
    print(f"\n{APP_NAME} — http://localhost:5000")
    print("API HeyGen:", "configurada" if HEYGEN_API_KEY else "não configurada")
    print("Wav2Lip:", WAV2LIP_DIR)
    print("SadTalker:", SADTALKER_DIR)
    print("MuseTalk:", MUSETALK_DIR)
    app.run(debug=False, host="0.0.0.0", port=5000)
