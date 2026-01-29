# core/views.py
import os
import time
import json
import requests
import hashlib
import re
import threading
from math import fabs
from typing import List, Dict, Any

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST, require_GET

from .forms import UrlForm
import trafilatura

# -------------------------
# Configuration / constants
# -------------------------
HF_API_URL = os.getenv("HF_API_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
print("Using HF model endpoint:", HF_API_URL)
print("Using HF token:", "set" if HF_TOKEN else "MISSING")

MIN_WORDS = 80                 # gate low-content pages
MAX_CHARS = 4000               # keep payload lean for latency

# Explainability settings
SENTENCE_LIMIT = 6             # how many sentences to test (controls API calls)
MAX_CHARS_EXPL = 3000          # char budget for explanation calls

# Simple in-memory cache (process-local)
EXPLANATION_CACHE: Dict[str, Dict[str, Any]] = {}

# -------------------------
# Low-level helpers
# -------------------------

def _fetch_article_text(url: str) -> str:
    """
    Fetch page and extract article-like text. Consider robots.txt checks in production.
    """
    r = requests.get(url, timeout=12, headers={"User-Agent": "NewsAnalyzer/1.0"})
    r.raise_for_status()
    text = trafilatura.extract(r.text, include_links=False, include_tables=False) or ""

    print(f"Fetched {len(r.content)} bytes from {url}, extracted {len(text)} chars of text")
    return text.strip()


def _pick_top_label(data):
    """
    Normalizes HF Serverless outputs to a single dict: {"label": str, "score": float}.
    Handles:
      - [{"label": "...", "score": 0.93}, ...]
      - [[{"label": "...", "score": 0.93}, {"label": "...", "score": 0.07}]]
    """
    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected HF response type: {type(data)} | {str(data)[:200]}")
    if isinstance(data[0], dict):
        return max(data, key=lambda x: x.get("score", 0.0))
    if isinstance(data[0], list) and data[0] and isinstance(data[0][0], dict):
        return max(data[0], key=lambda x: x.get("score", 0.0))
    raise ValueError(f"Unrecognized HF response shape: {str(data)[:200]}")


def _classify_remote(text: str, retries: int = 1, backoff: float = 1.5) -> dict:
    """
    Call HF Serverless inference with truncation/padding parameters. Returns a dict with label/score.
    """
    clean = " ".join((text or "").split())
    if not clean:
        raise ValueError("No text to classify after cleaning.")

    payload = {
        "inputs": clean,
        "parameters": {
            "truncation": True,
            "max_length": 512,
            "padding": "max_length",
            "top_k": 1,
            "return_all_scores": False
        },
        "options": {"wait_for_model": True}
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

    last_err = None
    for attempt in range(retries + 1):
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code in (503, 529, 429) and attempt < retries:
            time.sleep(backoff ** attempt)
            continue
        if not resp.ok:
            last_err = resp.text[:600]
            if resp.status_code == 400 and last_err:
                raise ValueError(f"Inference API rejected the request: {last_err}")
            resp.raise_for_status()
        data = resp.json()
        return _pick_top_label(data)
    raise RuntimeError(f"Inference failed. Last error: {last_err or 'None'}")


def _normalize_label(label: str) -> str:
    l = (label or "").strip().upper()
    if l in {"FAKE", "LABEL_0", "NEGATIVE"}:
        return "FAKE"
    if l in {"REAL", "LABEL_1", "POSITIVE"}:
        return "REAL"
    return label


# -------------------------
# Explainability helpers
# -------------------------

def split_into_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents


def fake_score_from_result(res: dict) -> float:
    lbl = (res.get("label") or "").upper()
    sc = float(res.get("score", 0.0))
    return sc if "FAKE" in lbl else (1.0 - sc)


def explain_by_occlusion_v2(article_text: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Improved occlusion-based explainability:
     - placeholder occlusion
     - sentence-alone scoring
     - combined ranking
    Returns base + sentence_impacts + explanation_text
    """
    content_hash = hashlib.sha256(article_text.encode("utf-8")).hexdigest()
    if content_hash in EXPLANATION_CACHE and EXPLANATION_CACHE[content_hash].get("status") == "ready":
        return EXPLANATION_CACHE[content_hash].get("payload")

    clean = " ".join(article_text.split())[:MAX_CHARS_EXPL]
    if not clean or len(clean.split()) < 20:
        payload = {"base": None, "sentence_impacts": [], "explanation_text": "Insufficient text to generate an explanation."}
        return payload

    try:
        base = _classify_remote(clean)
    except Exception as e:
        return {"base": None, "sentence_impacts": [], "explanation_text": f"Could not compute base prediction: {e}"}

    base_fake = fake_score_from_result(base)
    sents = split_into_sentences(clean)
    if not sents:
        return {"base": base, "sentence_impacts": [], "explanation_text": "No sentences found for explanation."}

    # Candidate set
    sents_to_test = sents[: max(len(sents), SENTENCE_LIMIT * 2)]
    impacts = []

    for sent in sents_to_test:
        placeholder = " [REDACTED] "
        occluded = clean.replace(sent, placeholder, 1).strip()
        if len(occluded.split()) < 20:
            continue
        try:
            occluded_res = _classify_remote(occluded)
            occluded_fake = fake_score_from_result(occluded_res)
        except Exception as e:
            print("Occlusion call failed:", e)
            continue
        delta = base_fake - occluded_fake

        sent_fake = None
        if len(sent.split()) >= 3:
            try:
                sent_res = _classify_remote(sent[:MAX_CHARS_EXPL])
                sent_fake = fake_score_from_result(sent_res)
            except Exception:
                sent_fake = None

        combined = abs(delta) + (0.5 * (sent_fake or 0.0))
        impacts.append({
            "sentence": sent,
            "delta": delta,
            "delta_pct": delta * 100.0,
            "sentence_fake": sent_fake,
            "combined": combined
        })

    impacts_sorted = sorted(impacts, key=lambda x: x["combined"], reverse=True)[:top_k]

    sentence_impacts = []
    for it in impacts_sorted:
        role = "increases_fake" if it["delta"] > 0 else "decreases_fake"
        sentence_impacts.append({
            "sentence": it["sentence"],
            "delta_raw": it["delta"],
            "delta_display": f"{it['delta_pct']:.2f}%",
            "sentence_fake": f"{(it['sentence_fake']*100):.2f}%" if it['sentence_fake'] is not None else "n/a",
            "role": role
        })

    if not sentence_impacts:
        explanation_text = "No high-impact sentences were identified. The model's prediction may be driven by global cues (style, source)."
    else:
        pieces = []
        for s in sentence_impacts:
            pieces.append(f"\"{s['sentence'][:140]}...\" => occlusion impact {s['delta_display']}, sentence-only {s['sentence_fake']}")
        explanation_text = "Top contributing sentences:\n" + "\n".join(pieces)

    payload = {"base": base, "sentence_impacts": sentence_impacts, "explanation_text": explanation_text}
    return payload


# -------------------------
# Background worker
# -------------------------

def _background_explain(article_text: str, content_hash: str):
    try:
        if content_hash in EXPLANATION_CACHE and EXPLANATION_CACHE[content_hash].get("status") == "ready":
            return
        EXPLANATION_CACHE[content_hash] = {"status": "pending"}
        explanation = explain_by_occlusion_v2(article_text, top_k=4)
        EXPLANATION_CACHE[content_hash] = {
            "status": "ready",
            "payload": {
                "explanation_text": explanation.get("explanation_text"),
                "sentence_impacts": explanation.get("sentence_impacts", []),
                "base": explanation.get("base"),
            }
        }
    except Exception as e:
        EXPLANATION_CACHE[content_hash] = {"status": "error", "error": str(e)}


# -------------------------
# Views / endpoints
# -------------------------

@csrf_protect
@csrf_protect
def classify_view(request):
    """
    Synchronous (non-AJAX) handler: GET shows form, POST performs full analyze and renders result.
    Note: synchronous explanation may add latency; consider keeping async worker for heavy work.
    """
    context = {"form": UrlForm()}

    if request.method == "POST":
        form = UrlForm(request.POST)
        context["form"] = form
        if form.is_valid():
            url = form.cleaned_data["url"]
            try:
                article_text = _fetch_article_text(url)
            except requests.RequestException as e:
                context["error"] = f"Failed to fetch URL: {e}"
                return render(request, "core/classify.html", context)

            if len(article_text.split()) < MIN_WORDS:
                context["error"] = "We could not extract sufficient article text from the URL."
                return render(request, "core/classify.html", context)

            # Base classification
            try:
                result = _classify_remote(article_text)
            except Exception as e:
                context["error"] = f"Inference error: {e}"
                return render(request, "core/classify.html", context)

            label_raw = result.get("label", "")
            score = float(result.get("score", 0.0))
            content_hash = hashlib.sha256(article_text.encode("utf-8")).hexdigest()

            context.update({
                "url": url,
                "label": _normalize_label(label_raw),
                "confidence": f"{score:.2%}",
                "model_id": HF_API_URL.replace("https://api-inference.huggingface.co/models/","") if HF_API_URL else "unknown",
                "hash": content_hash,
                "disclaimer": "Automated prediction; treat as decision support.",
            })

            # Synchronous explanation (optional - may be slow)
            try:
                explanation = explain_by_occlusion_v2(article_text, top_k=4)
                context["explanation_text"] = explanation.get("explanation_text")
                context["sentence_impacts"] = explanation.get("sentence_impacts", [])
            except Exception as ex:
                # do not block the result on explainability
                context["explanation_text"] = "Explanation could not be generated at this time."
                context["sentence_impacts"] = []

        else:
            context["error"] = "Invalid URL."

    return render(request, "core/classify.html", context)


@require_POST
def classify_ajax(request):
    """
    AJAX analyze endpoint. Returns base prediction JSON and triggers background explain.
    """
    if request.headers.get("x-requested-with") != "XMLHttpRequest":
        return JsonResponse({"error": "Invalid request"}, status=400)

    url = request.POST.get("url")
    if not url:
        return JsonResponse({"error": "URL missing"}, status=400)

    try:
        article_text = _fetch_article_text(url)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Failed to fetch URL: {e}"}, status=400)

    if len(article_text.split()) < MIN_WORDS:
        return JsonResponse({"error": "Could not extract sufficient article text from the URL."}, status=400)

    try:
        base_result = _classify_remote(article_text)
    except Exception as e:
        return JsonResponse({"error": f"Inference error: {e}"}, status=500)

    label_raw = base_result.get("label", "")
    score = float(base_result.get("score", 0.0))
    content_hash = hashlib.sha256(article_text.encode("utf-8")).hexdigest()

    resp = {
        "label": _normalize_label(label_raw),
        "confidence": f"{score:.2%}",
        "model_id": HF_API_URL.replace("https://api-inference.huggingface.co/models/","") if HF_API_URL else "unknown",
        "hash": content_hash,
        "disclaimer": "Automated prediction via Hugging Face Serverless Inference API; treat as decision support, not definitive fact-check.",
    }

    t = threading.Thread(target=_background_explain, args=(article_text, content_hash), daemon=True)
    t.start()

    return JsonResponse(resp)


@require_GET
def explain_status(request):
    """
    Polling endpoint: ?hash=<content_hash>
    """
    content_hash = request.GET.get("hash")
    if not content_hash:
        return JsonResponse({"error": "hash missing"}, status=400)

    payload = EXPLANATION_CACHE.get(content_hash)
    if not payload:
        return JsonResponse({"status": "pending"})

    if payload.get("status") == "pending":
        return JsonResponse({"status": "pending"})
    if payload.get("status") == "error":
        return JsonResponse({"status": "error", "error": payload.get("error")})

    # ready
    pd = payload.get("payload", {})
    return JsonResponse({
        "status": "ready",
        "explanation_text": pd.get("explanation_text"),
        "sentence_impacts": pd.get("sentence_impacts", []),
    })
