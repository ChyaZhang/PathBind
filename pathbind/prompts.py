"""Prompt templates used to evaluate PathBind.

Non-thinking models (e.g., LLaVA-1.5 family) may strip the <think>...</think>
instruction; only the <answer>X</answer> tag is required for scoring.
"""

# ---------- Multiple-choice VQA (PathBind-VQA D1--D6, PathBind-PTA D1--D5) ----------
MCQ_SYSTEM = (
    "You are a pathology expert. Your task is to answer the multiple-choice question step "
    "by step. Carefully observe the histological features in the provided pathology image, "
    "recall related pathology knowledge, and reason through the answer.\n"
    "In your reasoning, **explicitly describe what you see in the image** — the cells, "
    "tissues, structures, staining patterns, spatial arrangement, and any annotations "
    "(arrows, labels, scale bars). Do not use phrases like 'mentioned', 'context', "
    "'described', 'given', or 'provided' — refer to information as being 'in the image'.\n"
    "Output your response in the following format strictly:\n"
    "<think>\n"
    "Your step-by-step reasoning. First describe what is visually observable in the image, "
    "then connect it to pathology knowledge.\n"
    "</think>\n"
    "<answer>X</answer>\n"
    "where X is the single letter (A, B, C, D, ...) of the correct option."
)

MCQ_USER = (
    "{question_with_options}\n"
    "Select the single best answer. Place ONLY the letter inside <answer>...</answer>. "
    "Your reasoning inside <think>...</think> should reference specific visual features "
    "you observe in the image."
)

# ---------- Yes/No VQA (Quilt-VQA, Path-VQA subsets carried into PathBind-VQA) ----------
YN_SYSTEM = (
    "You are a pathology expert. Your task is to answer a yes/no question step by step. "
    "Carefully observe the histological features in the provided pathology image, recall "
    "related pathology knowledge, and reason through the answer.\n"
    "In your reasoning, **explicitly describe what you see in the image** — the cells, "
    "tissues, structures, staining patterns, spatial arrangement, and any annotations "
    "(arrows, labels, scale bars). Do not use phrases like 'mentioned', 'context', "
    "'described', 'given', or 'provided' — refer to information as being 'in the image'.\n"
    "Output your response in the following format strictly:\n"
    "<think>\n"
    "Your step-by-step reasoning. First describe what is visually observable in the image, "
    "then connect it to pathology knowledge.\n"
    "</think>\n"
    "<answer>Yes</answer> or <answer>No</answer>"
)

YN_USER = (
    "{question}\n"
    "Answer with Yes or No. Place ONLY 'Yes' or 'No' inside <answer>...</answer>. "
    "Your reasoning inside <think>...</think> should reference specific visual features "
    "you observe in the image."
)

# ---------- Entity-attention extraction (PathBind-Grounding) ----------
# Minimal prompt: no answer, just a phrase for the model to attend to.
ATTN_PROMPT = "Look at the image and locate: {entity}"


def format_options(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOP"
    return "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(options))


def build_mcq_messages(question: str, options: list[str], image_path: str | None):
    """OpenAI-style chat messages. `image_path=None` gives the text-only variant."""
    q = f"{question}\n{format_options(options)}"
    user_content: list = [{"type": "text", "text": MCQ_USER.format(question_with_options=q)}]
    if image_path is not None:
        user_content.insert(0, {"type": "image", "image": image_path})
    return [
        {"role": "system", "content": MCQ_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_yn_messages(question: str, image_path: str | None):
    user_content: list = [{"type": "text", "text": YN_USER.format(question=question)}]
    if image_path is not None:
        user_content.insert(0, {"type": "image", "image": image_path})
    return [
        {"role": "system", "content": YN_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ---------- Answer parsing ----------
import re
_ANSWER_RE = re.compile(r"<answer>\s*([A-Pa-p]|yes|no|Yes|No|YES|NO)\s*</answer>")


def parse_answer(raw: str, answer_type: str = "mcq") -> str | None:
    """Return the parsed letter (upper for mcq) / 'yes'/'no' (lower for yn), or None."""
    if not raw:
        return None
    m = _ANSWER_RE.search(raw)
    if m:
        token = m.group(1)
        return token.upper() if answer_type == "mcq" else token.lower()
    # Fallback: first standalone letter or Yes/No in the last 40 chars
    tail = raw.strip()[-40:]
    if answer_type == "mcq":
        m2 = re.search(r"\b([A-P])\b", tail)
        return m2.group(1) if m2 else None
    m2 = re.search(r"\b(Yes|No|yes|no)\b", tail)
    return m2.group(1).lower() if m2 else None
