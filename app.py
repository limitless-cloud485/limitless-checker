"""Limitless footage checker — visual verification service for the web finder.

Receives a narration line + candidate clip URLs, downloads each clip,
samples real frames, CLIP-scores them against the line, and returns a
ranked verdict (strong / ok / weak) per clip. Free-tier friendly: CPU
ONNX (Xenova CLIP ViT-B/32 quantized), small downloads, hard caps.

API (Gradio): verify(line: str, candidates_json: str) -> JSON string
  candidates_json: [{"url": "...", "title": "..."}, ...]  (max 8 used)
  returns: [{"url","title","score","neg","band","error"?}] sorted by score
"""
import io
import json
import os
import subprocess
import tempfile

import numpy as np
import requests

MAX_CANDIDATES = 8
MAX_BYTES = 80 * 1024 * 1024
DL_TIMEOUT = 60

NEGATIVES = [
    "a cartoon animation",
    "a website or news screenshot",
    "a presenter talking to the camera",
    "an orbit line diagram or chart",
    "a title card with text",
    "a city street in the rain",
    "people watching the night sky",
    "a star chart map",
]

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


class CLIP:
    def __init__(self, model_dir):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        kw = {"providers": ["CPUExecutionProvider"]}
        self.vision = ort.InferenceSession(os.path.join(model_dir, "vision_model_quantized.onnx"), **kw)
        self.textm = ort.InferenceSession(os.path.join(model_dir, "text_model_quantized.onnx"), **kw)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_padding(length=77)
        self.tok.enable_truncation(max_length=77)
        self.tin = {i.name for i in self.textm.get_inputs()}

    def text_embed(self, text):
        enc = self.tok.encode(text)
        feed = {"input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        feed = {k: v for k, v in feed.items() if k in self.tin}
        out = self.textm.run(None, feed)
        return _norm(out[0][0].astype(np.float32))

    def image_embed(self, pil_img):
        from PIL import Image
        img = pil_img.convert("RGB")
        w, h = img.size
        s = 224 / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
        w, h = img.size
        img = img.crop(((w - 224) // 2, (h - 224) // 2, (w + 224) // 2, (h + 224) // 2))
        arr = (np.asarray(img).astype(np.float32) / 255.0 - MEAN) / STD
        out = self.vision.run(None, {"pixel_values": arr.transpose(2, 0, 1)[None].astype(np.float32)})
        return _norm(out[0][0].astype(np.float32))


_clip = None


def get_clip():
    global _clip
    if _clip is None:
        model_dir = os.environ.get("CLIP_DIR")
        if not model_dir or not os.path.exists(os.path.join(model_dir, "vision_model_quantized.onnx")):
            from huggingface_hub import hf_hub_download
            base = hf_hub_download("Xenova/clip-vit-base-patch32", "tokenizer.json")
            model_dir = os.path.dirname(base)
            for f in ("vision_model_quantized.onnx", "text_model_quantized.onnx"):
                p = hf_hub_download("Xenova/clip-vit-base-patch32", f"onnx/{f}")
                dst = os.path.join(model_dir, f)
                if not os.path.exists(dst):
                    os.link(p, dst) if os.path.dirname(p) == model_dir else None
                    if not os.path.exists(dst):
                        import shutil
                        shutil.copy(p, dst)
        _clip = CLIP(model_dir)
    return _clip


def _download(url, path):
    with requests.get(url, stream=True, timeout=DL_TIMEOUT,
                      headers={"User-Agent": "LimitlessChecker/1.0"}) as r:
        r.raise_for_status()
        n = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                n += len(chunk)
                if n > MAX_BYTES:
                    raise ValueError("clip too large")
                f.write(chunk)


def _frames(path, tmp):
    from PIL import Image
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, timeout=30)
    try:
        d = float(dur.stdout.strip())
    except Exception:
        d = 0.0
    stamps = [d * f for f in (0.2, 0.4, 0.6, 0.8)] if d > 2 else [0.5, 1.5, 2.5, 3.5]
    imgs = []
    for i, t in enumerate(stamps):
        out = os.path.join(tmp, f"f{i}.jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path,
                        "-frames:v", "1", "-vf", "scale=320:-2", out, "-y"],
                       capture_output=True, timeout=60)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            imgs.append(Image.open(out).copy())
    return imgs


def _band(score, neg):
    if score >= 0.30 and score > neg + 0.005:
        return "strong"
    if score >= 0.22 and score > neg:
        return "ok"
    return "weak"


def verify(line, candidates_json):
    clip = get_clip()
    words = (line or "").split()
    desc = " ".join(words[:16]) if words else ""
    if not desc:
        return json.dumps({"error": "empty line"})
    try:
        cands = json.loads(candidates_json or "[]")
    except Exception:
        return json.dumps({"error": "bad candidates json"})
    cands = [c for c in cands if c.get("url")][:MAX_CANDIDATES]
    temb = clip.text_embed(desc)
    neg_embs = [clip.text_embed(t) for t in NEGATIVES]
    results = []
    for c in cands:
        row = {"url": c["url"], "title": c.get("title", "")}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "clip.mp4")
                _download(c["url"], path)
                imgs = _frames(path, tmp)
                if not imgs:
                    raise ValueError("no frames extractable")
                scores, negs = [], []
                for img in imgs:
                    ie = clip.image_embed(img)
                    scores.append(float(np.dot(temb, ie)))
                    negs.append(max(float(np.dot(ne, ie)) for ne in neg_embs))
                row["score"] = round(float(np.mean(scores)), 3)
                row["neg"] = round(float(np.mean(negs)), 3)
                row["band"] = _band(row["score"], row["neg"])
        except Exception as e:
            row["error"] = str(e)[:120]
            row["band"] = "error"
            row["score"] = -1
        results.append(row)
    results.sort(key=lambda r: -r["score"])
    return json.dumps({"line": desc, "results": results})


if __name__ == "__main__" or os.environ.get("SPACE_ID"):
    import gradio as gr
    demo = gr.Interface(
        fn=verify,
        inputs=[gr.Textbox(label="Narration line"),
                gr.Textbox(label="Candidates JSON", lines=4)],
        outputs=gr.Textbox(label="Ranked results JSON"),
        title="Limitless Footage Checker",
        description="Visual verification for the Limitless footage finder: scores candidate clips against a narration line using real frames.",
    )
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
