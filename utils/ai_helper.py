"""
ai_helper.py
─────────────────────────────────────────────
IBM watsonx.ai helper – all API calls go through here.

Model: mistralai/mistral-small-3-1-24b-instruct-2503
Auth:  IAM API key → short-lived Bearer token (cached for ~1 hour)
"""

import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Works in both local (.env) and Streamlit Cloud (Secrets)
try:
    WATSONX_API_KEY = st.secrets["WATSONX_API_KEY"].strip()
    WATSONX_PROJECT_ID = st.secrets["WATSONX_PROJECT_ID"].strip()
    WATSONX_REGION = st.secrets.get("WATSONX_REGION", "eu-gb").strip()
except Exception:
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "").strip()
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "").strip()
    WATSONX_REGION = os.getenv("WATSONX_REGION", "eu-gb").strip()

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
WATSONX_URL = f"https://{WATSONX_REGION}.ml.cloud.ibm.com"

MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"

# Learner level context
_LEVEL_CONTEXT = {
    "Beginner": (
        "a complete beginner with no prior knowledge of the subject. "
        "Use very simple words, short sentences, relatable examples, and avoid technical jargon."
    ),
    "Intermediate": (
        "a student who has basic familiarity with the subject. "
        "Use moderate technical vocabulary, explain specialised terms briefly, and connect ideas clearly."
    ),
    "Expert": (
        "an advanced learner or domain expert. "
        "Use precise technical language, deeper explanation, and advanced detail where relevant."
    ),
}


# Maximum characters of user content sent to IBM watsonx in a single prompt.
# 3,000 chars ≈ 750 tokens — keeps total prompt under ~1,000 tokens so the
# 24B Mistral model on the Lite plan responds well within 60 s.
MAX_INPUT_CHARS = 2000


def _clip_text(text: str) -> tuple[str, bool]:
    """Return (clipped_text, was_clipped).
    Clips to MAX_INPUT_CHARS at the nearest sentence boundary where possible.
    """
    if len(text) <= MAX_INPUT_CHARS:
        return text, False
    clipped = text[:MAX_INPUT_CHARS]
    # Try to end at a sentence boundary so the prompt isn't mid-sentence.
    last_period = max(clipped.rfind(". "), clipped.rfind(".\n"))
    if last_period > MAX_INPUT_CHARS * 0.7:
        clipped = clipped[: last_period + 1]
    return clipped, True


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

# IAM token cache: avoids a fresh HTTP round-trip on every button click.
# Tokens are valid for 3600 s; we refresh 60 s before expiry.
_token_cache: dict = {}


def _get_iam_token() -> str:
    """Exchange IBM Cloud API key for a short-lived IAM Bearer token.
    Result is cached in module memory and reused until ~60 s before expiry.
    """
    now = time.time()
    if _token_cache.get("token") and _token_cache.get("expires_at", 0) > now + 60:
        return _token_cache["token"]

    response = requests.post(
        IAM_TOKEN_URL,
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WATSONX_API_KEY,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"IAM token request failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    _token_cache["token"]      = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 3600)
    return _token_cache["token"]


def _generate(prompt: str, max_tokens: int = 400) -> str:
    """Send prompt to IBM watsonx.ai and return generated text."""
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return (
            "⚠️ Configuration missing — please add WATSONX_API_KEY and "
            "WATSONX_PROJECT_ID in your .env file."
        )

    try:
        token = _get_iam_token()

        url = f"{WATSONX_URL}/ml/v1/text/generation?version=2024-05-31"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        body = {
            "model_id": MODEL_ID,
            "input": prompt,
            "project_id": WATSONX_PROJECT_ID,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_tokens,
                "min_new_tokens": 10,
                "repetition_penalty": 1.05,
            },
        }

        response = requests.post(url, headers=headers, json=body, timeout=120)

        if response.status_code != 200:
            return f"⚠️ IBM watsonx error ({response.status_code})\n\n{response.text}"

        result = response.json()

        if "results" in result and result["results"]:
            return result["results"][0].get("generated_text", "").strip()

        return "⚠️ IBM watsonx returned an empty result."

    except Exception as exc:
        return f"⚠️ IBM watsonx error: {exc}"


# ─────────────────────────────────────────────────────────────
# Agent upgrade feature 1: Auto detect learner level
# ─────────────────────────────────────────────────────────────

def auto_detect_level(text: str) -> str:
    """Detect whether the content is best suited for Beginner, Intermediate, or Expert.
    Returns only one word: Beginner / Intermediate / Expert
    """
    text, _ = _clip_text(text)
    prompt = (
        "You are an educational AI classifier.\n"
        "Read the following academic content and decide what learner level is most appropriate.\n\n"
        "Choose ONLY one from these options:\n"
        "- Beginner\n"
        "- Intermediate\n"
        "- Expert\n\n"
        "Return only the level name and nothing else.\n\n"
        f"Content:\n{text}\n\n"
        "Level:"
    )
    result = _generate(prompt, 20).strip()
    if "Expert" in result:
        return "Expert"
    elif "Intermediate" in result:
        return "Intermediate"
    else:
        return "Beginner"


# ─────────────────────────────────────────────────────────────
# Existing public features
# Each function returns (result_text, was_clipped) so the UI
# can show an info notice when the content was truncated.
# ─────────────────────────────────────────────────────────────

def simplify_content(text: str, level: str = "Beginner") -> str:
    text, _ = _clip_text(text)
    ctx = _LEVEL_CONTEXT.get(level, _LEVEL_CONTEXT["Beginner"])
    prompt = (
        f"You are an expert teacher adapting academic material for {ctx}\n\n"
        "Rewrite the following academic content so it is perfectly suited to that audience. "
        "Keep all key ideas but adjust vocabulary, depth, and style to match the level.\n\n"
        f"Academic Content:\n{text}\n\n"
        "Rewritten Explanation:"
    )
    return _generate(prompt, 200)


def generate_summary(text: str, level: str = "Beginner") -> str:
    text, _ = _clip_text(text)
    ctx = _LEVEL_CONTEXT.get(level, _LEVEL_CONTEXT["Beginner"])
    prompt = (
        f"You are summarising academic content for {ctx}\n\n"
        "Summarise the following content in 5 to 7 clear bullet points. "
        "Be concise. Adjust depth to suit the audience level.\n\n"
        f"Content:\n{text}\n\n"
        "Summary (bullet points):"
    )
    return _generate(prompt, 150)


def generate_mcqs(text: str, level: str = "Beginner") -> str:
    text, _ = _clip_text(text)
    ctx = _LEVEL_CONTEXT.get(level, _LEVEL_CONTEXT["Beginner"])
    prompt = (
        f"You are creating a quiz for {ctx}\n\n"
        "Generate exactly 5 multiple-choice questions from the following academic content.\n"
        "Each question must have 4 options: A, B, C, D. State the correct answer after each question.\n\n"
        f"Content:\n{text}\n\n"
        "MCQs:"
    )
    return _generate(prompt, 220)


def generate_exam_questions(text: str, level: str = "Beginner") -> str:
    text, _ = _clip_text(text)
    ctx = _LEVEL_CONTEXT.get(level, _LEVEL_CONTEXT["Beginner"])
    prompt = (
        f"You are setting exam questions for {ctx}\n\n"
        "Generate 5 important exam questions from the following academic content. "
        "Mix short-answer and descriptive questions.\n\n"
        f"Content:\n{text}\n\n"
        "Exam Questions:"
    )
    return _generate(prompt, 180)


def explain_keywords(text: str, level: str = "Beginner") -> str:
    text, _ = _clip_text(text)
    ctx = _LEVEL_CONTEXT.get(level, _LEVEL_CONTEXT["Beginner"])
    prompt = (
        f"You are explaining technical vocabulary to {ctx}\n\n"
        "Identify 5 important keywords from the following academic content. "
        "For each, give a one-sentence explanation.\n"
        "Format: Keyword - explanation\n\n"
        f"Content:\n{text}\n\n"
        "Keywords and Explanations:"
    )
    return _generate(prompt, 150)


def answer_question(text: str, question: str, level: str = "Beginner") -> str:
    text, _ = _clip_text(text)
    ctx = _LEVEL_CONTEXT.get(level, _LEVEL_CONTEXT["Beginner"])
    prompt = (
        f"You are a tutor answering a question for {ctx}\n\n"
        "Use only the provided content. If the answer is not in the content, say so.\n\n"
        f"Content:\n{text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return _generate(prompt, 150)


# ─────────────────────────────────────────────────────────────
# Agent upgrade feature 2: Study Plan Generator
# ─────────────────────────────────────────────────────────────

def generate_study_plan(text: str, level: str = "Beginner") -> str:
    """Create a smart study roadmap from the uploaded academic content."""
    text, _ = _clip_text(text)
    ctx = _LEVEL_CONTEXT.get(level, _LEVEL_CONTEXT["Beginner"])
    prompt = (
        f"You are an academic learning planner helping {ctx}\n\n"
        "Create a concise study plan from the following academic content.\n"
        "Include: main topics, recommended study order, what to revise first, and a short checklist.\n\n"
        f"Academic Content:\n{text}\n\n"
        "Study Plan:"
    )
    return _generate(prompt, 180)


# ─────────────────────────────────────────────────────────────
# Agent upgrade feature 3: Difficult concepts detector
# ─────────────────────────────────────────────────────────────

def find_difficult_concepts(text: str, level: str = "Beginner") -> str:
    """Identify difficult concepts, weak areas, or confusion points in the content."""
    text, _ = _clip_text(text)
    ctx = _LEVEL_CONTEXT.get(level, _LEVEL_CONTEXT["Beginner"])
    prompt = (
        f"You are an educational mentor helping {ctx}\n\n"
        "Identify up to 4 difficult concepts in the following academic content.\n"
        "For each: name it, explain why it's confusing, give a simple explanation, and one tip.\n\n"
        f"Academic Content:\n{text}\n\n"
        "Difficult Concepts:"
    )
    return _generate(prompt, 180)


# ─────────────────────────────────────────────────────────────
# CLI utility — run directly to list available models
# ─────────────────────────────────────────────────────────────

def list_models():
    token = _get_iam_token()
    url = f"{WATSONX_URL}/ml/v1/foundation_model_specs?version=2024-05-31"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
        return
    data = response.json()
    print("=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)
    for model in data["resources"]:
        print(model["model_id"])


if __name__ == "__main__":
    list_models()
