# agents/speech_agent.py
import json
from datetime import datetime
from typing import Dict, Any, Optional
import time
import uuid
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from utils.load_model import load_chat_model
from tools.memory_tools import db_get_thread, db_update_thread_fields, db_get_user_profile
from agents.score_agent import score_step


speech_model = load_chat_model(
    "google_vertexai/gemini-2.5-flash",
    tags=["speech_agent"],
    temperature=0.2,
)

SPEECH_AGENT_PROMPT = """
Bạn là SPEECH_AGENT cho chatbot luyện nghe/nói tiếng Anh cho người khiếm thị.

Bạn sẽ nhận input JSON tối thiểu:
{"user_id":"...","thread_id":"..."}

QUY TẮC:
- Chỉ giảng 1 bước ngắn (TTS-friendly).
- Kết thúc bằng yêu cầu người học phản hồi.
- KHÔNG bịa nội dung, chỉ dùng dữ liệu có trong DB.
- Luôn dùng đúng user_id và thread_id từ input.
"""

speech_agent = create_react_agent(
    model=speech_model,
    tools=[db_get_thread, db_update_thread_fields, db_get_user_profile],
    prompt=SystemMessage(content=SPEECH_AGENT_PROMPT),
    name="speech_agent_llm",
)
import re
PHASE_ORDER = ["passage", "listening_practice", "speaking_practice", "key_vocab", "evaluation_material"]
_REPEAT_INTENT_PATTERNS = [
    r"^(repeat|again)$",
    r"^repeat (it|please)$",
    r"^(can|could) you (please )?(repeat|say that again)\??$",
    r"^(please )?(repeat|say that again)\??$",
    r"^(đọc|nhắc|nói|lặp)\s+lại(\s+được\s+không|\s+giúp\s+mình|\s+nha|\?)?$",
    r"^bạn (đọc|nhắc|nói)\s+lại(\s+được\s+không|\s+nha|\?)?$",
]

def _is_repeat_cmd(text: Optional[str]) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(re.search(pat, t) for pat in _REPEAT_INTENT_PATTERNS)

_REPEAT_PASSAGE_PATTERNS = [
    r"\bpassage\b",
    r"\bcontent\b",
    r"\bdoan( nghe)?\b",
    r"\bbai doc\b",
    r"doc lai( doan| bai| passage| content)?",
    r"nhac lai( doan| bai| passage| content)?",
    r"noi lai( doan| bai| passage| content)?",
    r"repeat( the)? (passage|content)",
    r"read( the)? passage( again)?",
    r"(doc|nhac|noi)\s+lai(\s+(passage|content|doan( nghe)?|bai( doc| nghe)?))?",
]

def _is_repeat_passage_cmd(text: Optional[str]) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"[!?.,;:]+$", "", t)
    if not t:
        return False
    return any(re.search(pat, t) for pat in _REPEAT_PASSAGE_PATTERNS)

_SKIP_RE = re.compile(r"^(tiếp|tiep|next|skip|bỏ qua|bo qua)$", re.I)


def _is_skip_cmd(text: Optional[str]) -> bool:
    return bool(_SKIP_RE.search((text or "").strip()))
def _extract_listen_target(en: str) -> str:
    """
    Nếu en chứa dạng 'Listen and repeat ...: <sentence>' thì chỉ lấy <sentence> để chấm.
    """
    if not en:
        return ""
    s = en.strip()
    low = s.lower()
    if "listen and repeat" in low and ":" in s:
        return s.split(":", 1)[1].strip()
    return s
def _advance_until_different(plan: Dict[str, Any], progress: Dict[str, Any], prev_unit: str, max_hops: int = 6):
    """
    Tiến progress cho tới khi unit text khác prev_unit (tránh plan bị trùng nội dung).
    """
    def _norm_text(t: str) -> str:
        base = " ".join(re.findall(r"[^\W_]+", (t or "").lower()))
        base = re.sub(r"\b(buoc|step)\b", "", base)
        base = re.sub(r"\b\d+\b", "", base)
        return " ".join(base.split())

    p = dict(progress)
    prev_norm = _norm_text(prev_unit or "")
    for _ in range(max_hops):
        if p.get("done"):
            return p, ""
        unit = _render_one_unit(plan, p)
        if unit.strip() and _norm_text(unit) != prev_norm:
            return p, unit
        p = _next_progress(dict(p), plan)
    # nếu vẫn trùng sau nhiều hop, trả về unit hiện tại để không treo
    return p, _render_one_unit(plan, p)

def _norm_unit_text(t: str) -> str:
    base = " ".join(re.findall(r"[^\W_]+", (t or "").lower()))
    base = re.sub(r"\b(buoc|step)\b", "", base)
    base = re.sub(r"\b\d+\b", "", base)
    return " ".join(base.split())

def _llm_coach_hint(
    phase: str,
    unit_text: str,
    expected: Dict[str, Any],
    user_text: str,
    attempts: int,
) -> str:
    """
    LLM tạo gợi ý dẫn dắt (không bịa).
    - Chỉ dùng dữ liệu trong unit_text + expected.
    - TTS-friendly, 1-3 câu, kết thúc bằng yêu cầu thử lại.
    """
    sys = SystemMessage(content=(
        "Bạn là gia sư tiếng Anh cho người khiếm thị. "
        "Nhiệm vụ: khi học viên trả lời SAI, hãy đưa gợi ý DẪN DẮT ngắn gọn, thân thiện. "
        "QUY TẮC: Không bịa nội dung ngoài dữ liệu được cung cấp. "
        "Không nói lan man. 1-3 câu. Kết thúc bằng yêu cầu học viên thử lại. "
        "Nếu attempts >= 2 thì cho gợi ý rõ hơn (ví dụ: đưa khung câu có chỗ trống, "
        "hoặc nhắc 3-5 từ đầu của câu đúng), nhưng tránh đọc lại toàn bộ đáp án nếu không cần."
    ))
    hm = HumanMessage(content=json.dumps({
        "phase": phase,
        "attempts": attempts,
        "user_answer": user_text,
        "expected": expected,        # chỉ là key info
        "current_step_text": unit_text  # text hiển thị từ DB/plan
    }, ensure_ascii=False))

    try:
        msg = speech_model.invoke([sys, hm])
        return (getattr(msg, "content", "") or "").strip()
    except Exception:
        # fallback an toan
        return "Chua dung lam. Ban thu lai nhe."

def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        return None

def _llm_check_key_vocab(expected: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    item = (expected.get("item") or "").strip()
    meaning = (expected.get("meaning_vi") or "").strip()
    example = (expected.get("example") or "").strip()
    if not item:
        return {"passed": False, "feedback": "Minh chua co tu vung de cham."}

    sys = SystemMessage(content=(
    "You are an English tutor grading a learner's sentence for a target vocabulary word.\n"
    "You MUST decide PASS/FAIL using ONLY the provided fields.\n"
    "\n"
    "PASS criteria (ALL required):\n"
    "1) The learner's output is a FULL sentence with SUBJECT + FINITE VERB (a verb that is conjugated / carries tense).\n"
    "   Accept any of these as a finite verb pattern:\n"
    "   - be-verb: am/is/are/was/were + complement\n"
    "   - modal: can/could/will/would/should/must + base verb\n"
    "   - do-support: do/does/did + base verb\n"
    "   - have: have/has/had + past participle\n"
    "   - simple verb: study/learn/like/need/want/go/works/learned, etc.\n"
    "2) The sentence uses the target vocabulary as a WORD (not just a substring) and with roughly correct meaning.\n"
    "   Accept inflections/related forms if meaning stays correct (e.g., educate/educational/educated for 'education').\n"
    "3) Sentence must be at least 4 words (after removing punctuation).\n"
    "\n"
    "FAIL if ANY of these:\n"
    "- Only a noun phrase / fragment (e.g., 'The grilled salmon', 'My education', 'In education').\n"
    "- Missing a finite verb (e.g., 'I education', 'Education important', 'To study education').\n"
    "- Only a clause starter (e.g., 'Because...', 'When...') without a main clause.\n"
    "- Does not use the target vocab at all.\n"
    "\n"
    "Output STRICT JSON ONLY (no extra text):\n"
    "{\"passed\": true/false, \"feedback\": \"...\"}\n"
    "Feedback rules:\n"
    "- Vietnamese WITHOUT diacritics (unaccented).\n"
    "- 1 short sentence. Friendly.\n"
    "- If FAIL, tell them what to fix (chu ngu + dong tu; hoac dua khung: 'I ... education ...').\n"
))
    hm = HumanMessage(content=json.dumps({
        "target_vocab": item,
        "meaning_vi": meaning,
        "example": example,
        "user_sentence": user_text or "",
    }, ensure_ascii=True))
    try:
        msg = speech_model.invoke([sys, hm])
        data = _parse_json_from_text((getattr(msg, "content", "") or "").strip()) or {}
        passed = bool(data.get("passed"))
        feedback = (data.get("feedback") or "").strip()
        if not feedback:
            feedback = "Dung roi." if passed else "Chua dung, ban dat lai cau nhe."
        return {"passed": passed, "feedback": feedback}
    except Exception:
        # fallback: simple contains
        raw = (user_text or "").lower()
        words = [w for w in raw.split() if w]
        has_vocab = item.lower() in raw
        verbs = {"am","is","are","was","were","be","been","being","do","does","did","have","has","had","can","could","will","would","should","may","might","must"}
        has_verb = any(w in verbs for w in words)
        passed = has_vocab and len(words) >= 4 and has_verb
        feedback = "Dung roi." if passed else "Chua dung, ban dat lai cau nhe."
        return {"passed": passed, "feedback": feedback}

def _llm_check_speaking_prompt(prompt_en: str, user_text: str) -> Dict[str, Any]:
    if not prompt_en:
        return {"passed": False, "feedback": "Minh chua co prompt de cham."}
    sys = SystemMessage(content=(
    "You are an English tutor grading whether the learner's response is relevant to the given speaking prompt.\n"
    "You MUST decide PASS/FAIL using ONLY the provided fields.\n"
    "\n"
    "PASS rules:\n"
    "1) The response must be ON-TOPIC for the prompt.\n"
    "2) If the prompt contains MULTIPLE requirements, the response must address ALL of them.\n"
    "   Example: 'Describe a movie ... and why you liked it' => must (a) describe a movie AND (b) give at least one reason.\n"
    "3) The response must have enough content: at least 2 sentences OR at least 20 words.\n"
    "4) Minor grammar mistakes are OK if meaning is clear.\n"
    "\n"
    "FAIL rules (any of these):\n"
    "- Off-topic or unrelated.\n"
    "- Answers only one part of a multi-part prompt.\n"
    "- Too short (less than 2 sentences AND less than 20 words), unless it clearly covers all required parts (rare).\n"
    "- Empty, nonsense, or mostly Vietnamese.\n"
    "\n"
    "Return STRICT JSON ONLY (no extra text):\n"
    "{\"passed\": true/false, \"feedback\": \"...\"}\n"
    "Feedback rules:\n"
    "- Vietnamese WITHOUT diacritics (unaccented).\n"
    "- 1 short friendly sentence.\n"
    "- If FAIL, say what is missing (vi du: 'can mo ta phim va noi ly do ban thich').\n"
))
    hm = HumanMessage(content=json.dumps({
        "prompt": prompt_en,
        "user_response": user_text or "",
    }, ensure_ascii=True))
    try:
        msg = speech_model.invoke([sys, hm])
        data = _parse_json_from_text((getattr(msg, "content", "") or "").strip()) or {}
        passed = bool(data.get("passed"))
        feedback = (data.get("feedback") or "").strip()
        if not feedback:
            feedback = "Dung roi." if passed else "Ban tra loi chua dung huong. Ban thu lai nhe."
        return {"passed": passed, "feedback": feedback}
    except Exception:
        # fallback: accept to avoid blocking
        return {"passed": True, "feedback": "Tot. Ban noi day du."}

def _first_sentence(text: str) -> str:
    if not text:
        return ""
    for sep in [".", "?", "!"]:
        if sep in text:
            return text.split(sep, 1)[0].strip() + sep
    return text.strip()


def _get_expected_for_step(plan: Dict[str, Any], progress: Dict[str, Any]) -> Dict[str, Any]:
    lesson = plan.get("lesson") or {}
    phase = progress.get("phase")
    idx = int(progress.get("step_idx", 0))

    if phase == "passage":
        passage = (lesson.get("passage") or {}).get("text") or ""
        return {"text": passage}

    if phase == "listening_practice":
        steps = lesson.get("listening_practice") or []
        s = steps[idx] if idx < len(steps) else {}
        en_raw = s.get("en") or ""
        target = _extract_listen_target(en_raw)
        return {"text": target}

    if phase == "speaking_practice":
        steps = lesson.get("speaking_practice") or []
        s = steps[idx] if idx < len(steps) else {}
        return {"prompt_en": s.get("prompt_en") or ""}

    if phase == "key_vocab":
        vocab = lesson.get("key_vocab") or []
        v = vocab[idx] if idx < len(vocab) else {}
        return {
            "item": v.get("item") or "",
            "meaning_vi": v.get("meaning_vi") or "",
            "example": v.get("example") or "",
        }

    if phase == "evaluation_material":
          ev = lesson.get("evaluation_material") or {}
          qs = ev.get("listening_questions") or []
          q = qs[idx] if idx < len(qs) else {}
          return {
              "type": q.get("type"),
              "answer_key": q.get("answer_key"),
              "choices": q.get("choices") or [],
          }

    return {}


def _expected_is_empty(expected: Dict[str, Any]) -> bool:
    for v in expected.values():
        if v:
            return False
    return True


def _get_plan_from_thread(thread_blob: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    current_day = int(thread_blob.get("current_day", 1))
    plan = thread_blob.get(f"last_plan_day_{current_day}") or thread_blob.get("last_plan")
    return plan


def _safe_int(val: Any, default: int = 1) -> int:
    try:
        return int(float(val))
    except Exception:
        return default


def _init_progress(thread_blob: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    day_index = _safe_int((plan.get("meta") or {}).get("day_index") or thread_blob.get("current_day", 1))
    return {
        "day_index": day_index,
        "phase": "passage",
        "step_idx": 0,
        "done": False,
        "awaiting_answer": False,
        "attempts": 0,
    }


def _next_progress(progress: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    """Tăng step/phase. done=true khi qua hết phase cuối."""
    phase = progress["phase"]
    idx = int(progress.get("step_idx", 0))

    lesson = plan.get("lesson") or {}

    def phase_len(ph: str) -> int:
        if ph == "passage":
            return 1
        if ph == "evaluation_material":
            ev = lesson.get("evaluation_material") or {}
            qs = ev.get("listening_questions") or []
            return max(1, len(qs))
        arr = lesson.get(ph) or []
        return len(arr)

    cur_len = phase_len(phase)
    if idx + 1 < cur_len:
        progress["step_idx"] = idx + 1
        return progress

    # hết phase -> chuyển phase tiếp theo
    try:
        p_i = PHASE_ORDER.index(phase)
    except ValueError:
        p_i = 0

    if p_i + 1 >= len(PHASE_ORDER):
        progress["done"] = True
        return progress

    progress["phase"] = PHASE_ORDER[p_i + 1]
    progress["step_idx"] = 0
    return progress


def _render_one_unit(plan: Dict[str, Any], progress: Dict[str, Any]) -> str:
    meta = plan.get("meta") or {}
    lesson = plan.get("lesson") or {}
    phase = progress["phase"]
    idx = int(progress.get("step_idx", 0))
    
    topic = meta.get("selected_topic") or "Giao tiếp hằng ngày"
    day = meta.get("day_index") or 1
    day = int(day)
    # 1) PASSAGE
    if phase == "passage":
        passage = (lesson.get("passage") or {}).get("text") or ""
        return (
            f"Hôm nay là ngày {day}, chủ đề {topic}.\n"
            f"Đầu tiên, mình đọc đoạn nghe. Bạn nghe nhé:\n\n"
            f"{passage}\n\n"
            f"Bây giờ bạn nói lại 1 câu bạn nhớ nhất (tiếng Anh) nhé."
        )

    # 2) LISTENING PRACTICE
    if phase == "listening_practice":
        steps = lesson.get("listening_practice") or []
        if idx >= len(steps):
            return "Mình hết phần luyện nghe rồi. Bạn nói 'tiếp' để sang phần tiếp theo nhé."
        s = steps[idx]
        vi = s.get("vi", "")
        en_raw = s.get("en", "")
        en_target = _extract_listen_target(en_raw)
        return (
            f"Luyện nghe – bước {idx+1}.\n"
            f"{vi}\n"
            f"English: {en_target}\n\n"
            f"Bạn nhắc lại câu tiếng Anh này nhé."
        )
    # 3) SPEAKING PRACTICE
    if phase == "speaking_practice":
        steps = lesson.get("speaking_practice") or []
        if idx >= len(steps):
            return "Mình hết phần luyện nói rồi. Bạn nói 'tiếp' để sang phần từ vựng nhé."
        s = steps[idx]
        vi = s.get("vi", "")
        p = s.get("prompt_en", "")
        return (
            f"Luyện nói – bước {idx+1}.\n"
            f"{vi}\n"
            f"Prompt: {p}\n\n"
            f"Bạn trả lời theo prompt này khoảng 20–30 giây nhé."
        )

    # 4) KEY VOCAB
    if phase == "key_vocab":
        vocab = lesson.get("key_vocab") or []
        if idx >= len(vocab):
            return "Xong từ vựng rồi. Bạn nói 'tiếp' để làm câu hỏi nhé."
        v = vocab[idx]
        return (
            f"Từ vựng {idx+1}.\n"
            f"{v.get('item','')} – {v.get('meaning_vi','')}\n"
            f"Ví dụ: {v.get('example','')}\n\n"
            f"Bạn đặt 1 câu mới với từ này nhé."
        )

    # 5) EVALUATION MATERIAL
    if phase == "evaluation_material":
        ev = lesson.get("evaluation_material") or {}
        qs = ev.get("listening_questions") or []

        if qs:
            q = qs[idx] if idx < len(qs) else qs[-1]
            print("[EVAL] idx=", idx, "len=", len(qs),
              "q_en=", (q.get("q_en") or "")[:80],
              "answer_key=", q.get("answer_key"))
            if q.get("type") == "multiple_choice":
                ch = q.get("choices") or []
                return (
                    f"Cau hoi nghe ({idx+1}/{len(qs)} – trac nghiem): {q.get('q_en','')}\n"
                    f"A. {ch[0] if len(ch)>0 else ''}\n"
                    f"B. {ch[1] if len(ch)>1 else ''}\n"
                    f"C. {ch[2] if len(ch)>2 else ''}\n\n"
                    f"D. {ch[3] if len(ch)>3 else ''}\n\n"
                    f"Ban chon A, B, C hay D?"
                )
            return (
                f"Câu hỏi nghe (ngắn): {q.get('q_en','')}\n\n"
                f"Bạn trả lời 1 câu ngắn nhé."
            )

        sp = (ev.get("speaking_prompt") or {}).get("prompt_en")
        if sp:
            return f"Speaking prompt: {sp}\n\nBạn nói khoảng 20–30 giây nhé."

        return "Mình chưa có câu hỏi đánh giá. Bạn nói 'tiếp' để kết thúc bài nhé."

    return "Bạn nói 'tiếp' để mình hướng dẫn phần tiếp theo nhé."

def _render_passage_unit(plan: Dict[str, Any], progress: Dict[str, Any]) -> str:
    meta = plan.get("meta") or {}
    lesson = plan.get("lesson") or {}
    topic = meta.get("selected_topic") or "Giao tiếp hàng ngày"
    day = int(meta.get("day_index") or 1)
    passage = (lesson.get("passage") or {}).get("text") or ""
    return (
        f"Minh nhac lai passage ngay {day}, chu de {topic}:\n\n"
        f"{passage}\n\n"
    )


def _latest_session_record(user_profile: Dict[str, Any], today: str) -> Dict[str, Any]:
    records = user_profile.get("session_records") or []
    latest_date = ""
    for r in records:
        d = str(r.get("date") or "")
        if d and d < today and d > latest_date:
            latest_date = d
    if not latest_date:
        return {}
    best = None
    best_score = -1
    best_key = (-1, -1)
    for r in records:
        if str(r.get("date") or "") != latest_date:
            continue
        try:
            score = int(r.get("overall_score", -1))
        except Exception:
            score = -1
        try:
            day = int(r.get("day_index", -1))
        except Exception:
            day = -1
        try:
            att = int(r.get("attempt", -1))
        except Exception:
            att = -1
        key = (day, att)
        if score > best_score or (score == best_score and key > best_key):
            best_score = score
            best_key = key
            best = r
    return best or {}

def _latest_profile_date(user_profile: Dict[str, Any]) -> str:
    records = user_profile.get("session_records") or []
    best = ""
    for r in records:
        d = str(r.get("date") or "")
        if d and d > best:
            best = d
    return best

def _is_new_usage_day(user_profile: Dict[str, Any], thread_blob: Dict[str, Any], today: str) -> bool:
    # Only treat as a new day once per calendar day.
    last_thread_date = str(thread_blob.get("last_usage_date") or "")
    if last_thread_date == today:
        return False
    last_date = _latest_profile_date(user_profile)
    return bool(last_date and last_date < today)

def _bump_usage_day(thread_blob: Dict[str, Any], today: str) -> Dict[str, Any]:
    last_date = str(thread_blob.get("last_usage_date") or "")
    count = int(thread_blob.get("usage_day_count") or 0)
    if last_date != today:
        count += 1
    return {"last_usage_date": today, "usage_day_count": count}

def speech_step(user_id: str, thread_id: str, user_text: Optional[str] = None) -> str:
    _call_id = uuid.uuid4().hex[:8]
    print(f"[CALL {_call_id}] start t={time.time():.3f} user_id={user_id} thread_id={thread_id} user_text={(user_text or '')!r}")
    thread_blob = db_get_thread.invoke({"user_id": user_id, "thread_id": thread_id}) or {}
    plan = _get_plan_from_thread(thread_blob)
    if not plan:
        return "Hien chua co bai hoc trong he thong. Minh can tao bai truoc, ban thu lai sau nhe."

    # Reset on new usage day (based on user_profile)
    today = datetime.utcnow().date().isoformat()
    user_profile = db_get_user_profile.invoke({"user_id": user_id}) or {}
    if _is_new_usage_day(user_profile, thread_blob, today):
        progress = _init_progress(thread_blob, plan)
        fields = {"speech_progress": progress}
        fields.update(_bump_usage_day(thread_blob, today))
        db_update_thread_fields.invoke({
            "user_id": user_id,
            "thread_id": thread_id,
            "fields_json": json.dumps(fields, ensure_ascii=False)
        })
        thread_blob = db_get_thread.invoke({"user_id": user_id, "thread_id": thread_id}) or {}

    progress = thread_blob.get("speech_progress")
    if not isinstance(progress, dict):
        progress = _init_progress(thread_blob, plan)
    else:
        plan_day = (plan.get("meta") or {}).get("day_index")
        if plan_day is not None and _safe_int(progress.get("day_index")) != _safe_int(plan_day):
            progress = _init_progress(thread_blob, plan)
    print(f"[CALL {_call_id}] progress(normalized)={progress}", flush=True)
    if progress.get("done"):
        last_rec = _latest_session_record(user_profile, today)
        profile_date = str(last_rec.get("date") or "")
        last_logged = thread_blob.get("last_session_logged") or {}
        thread_date = str(last_logged.get("date") or "")
        if (profile_date and profile_date != today) or (thread_date and thread_date != today):
            progress = _init_progress(thread_blob, plan)
            db_update_thread_fields.invoke({
                "user_id": user_id,
                "thread_id": thread_id,
                "fields_json": json.dumps({"speech_progress": progress}, ensure_ascii=False)
            })
        else:
            return "Bai hom nay da ket thuc. Minh dang tinh diem, ban cho minh mot chut nhe."

    # ===== A) Nếu đang chờ trả lời =====
    if progress.get("awaiting_answer"):
        if not user_text or not str(user_text).strip():
            return "Minh dang cho cau tra loi. Ban noi lai nhe."

        # repeat passage: đọc lại passage, KHÔNG chấm, KHÔNG đổi progress
        if _is_repeat_passage_cmd(user_text):
            unit = _render_passage_unit(plan, progress)
            return f"Duoc nhe. Minh doc lai passage:\n\n{unit}"

        # repeat: chỉ đọc lại step, KHÔNG chấm, KHÔNG đổi progress
        if _is_repeat_cmd(user_text):
            unit = _render_one_unit(plan, progress)
            return f"Duoc nhe. Minh nhac lai buoc nay:\n\n{unit}"

        # skip/tiếp: nhảy bước
        if _is_skip_cmd(user_text):
            new_progress = _next_progress(dict(progress), plan)
            new_progress["awaiting_answer"] = True
            new_progress["attempts"] = 0
            db_update_thread_fields.invoke({
                "user_id": user_id,
                "thread_id": thread_id,
                "fields_json": json.dumps({"speech_progress": new_progress}, ensure_ascii=False)
            })
            if new_progress.get("done"):
                return "Bai hom nay da ket thuc. Minh dang tinh diem, ban cho minh mot chut nhe."
            return _render_one_unit(plan, new_progress)

        expected = _get_expected_for_step(plan, progress)
        prev_unit = _render_one_unit(plan, progress)
        if progress.get("phase") == "evaluation_material":
            ev = (plan.get("lesson") or {}).get("evaluation_material") or {}
            qs = ev.get("listening_questions") or []
            idx0 = int(progress.get("step_idx", 0))
            q_en0 = ""
            if qs and 0 <= idx0 < len(qs):
                q_en0 = (qs[idx0].get("q_en") or "")
            print(f"[CALL {_call_id}] EVAL(before_score) idx={idx0} len={len(qs)} q_en={(q_en0[:120])!r}")
        # expected rỗng -> tự động chuyển bước, SAVE + RETURN NGAY
        if _expected_is_empty(expected):
            new_progress = _next_progress(dict(progress), plan)
            new_progress["awaiting_answer"] = True
            new_progress["attempts"] = 0
            db_update_thread_fields.invoke({
                "user_id": user_id,
                "thread_id": thread_id,
                "fields_json": json.dumps({"speech_progress": new_progress}, ensure_ascii=False)
            })
            if new_progress.get("done"):
                return "Bai hom nay da ket thuc. Minh dang tinh diem, ban cho minh mot chut nhe."
            return _render_one_unit(plan, new_progress)

        # ===== B) Chấm điểm =====
        if progress.get("phase") == "key_vocab":
            result = _llm_check_key_vocab(expected, user_text or "")
        elif progress.get("phase") == "speaking_practice":
            prompt_en = expected.get("prompt_en") or ""
            word_count = len((user_text or "").split())
            if prompt_en and word_count > 6:
                result = _llm_check_speaking_prompt(prompt_en, user_text or "")
            else:
                result = score_step(progress.get("phase", ""), expected, user_text)
        else:
            result = score_step(progress.get("phase", ""), expected, user_text)

        # sai -> tăng attempts + LLM hint, SAVE progress, RETURN step hiện tại
        if not result.get("passed"):
            new_progress = dict(progress)
            new_progress["attempts"] = int(new_progress.get("attempts", 0)) + 1
            attempts = int(new_progress["attempts"])

            unit = _render_one_unit(plan, new_progress)
            hint = _llm_coach_hint(
                phase=new_progress.get("phase", ""),
                unit_text=unit,
                expected=expected,
                user_text=user_text,
                attempts=attempts,
            )

            db_update_thread_fields.invoke({
                "user_id": user_id,
                "thread_id": thread_id,
                "fields_json": json.dumps({"speech_progress": new_progress}, ensure_ascii=False)
            })
            return f"{hint}\n\n{unit}"

        # đúng -> chuyển bước bằng _next_progress, SAVE + RETURN step mới
        new_progress = _next_progress(dict(progress), plan)
        print(f"[CALL {_call_id}] ADVANCE {progress.get('phase')}:{progress.get('step_idx')} -> {new_progress.get('phase')}:{new_progress.get('step_idx')}")

        new_progress["awaiting_answer"] = True
        new_progress["attempts"] = 0

        # ✅ skip unit trùng
        
        print(f"[CALL {_call_id}] BEFORE_DEDUP phase={new_progress.get('phase')} idx={new_progress.get('step_idx')}", flush=True)
        new_progress, next_unit = _advance_until_different(plan, new_progress, prev_unit)
        print(f"[CALL {_call_id}] AFTER_DEDUP phase={new_progress.get('phase')} idx={new_progress.get('step_idx')}", flush=True)
        if (
            not new_progress.get("done")
            and _norm_unit_text(next_unit) == _norm_unit_text(prev_unit)
        ):
            # hop further to find a different step; if none, end
            hopped = False
            tmp = dict(new_progress)
            for _ in range(4):
                tmp = _next_progress(dict(tmp), plan)
                if tmp.get("done"):
                    new_progress = tmp
                    next_unit = ""
                    hopped = True
                    break
                unit = _render_one_unit(plan, tmp)
                if _norm_unit_text(unit) != _norm_unit_text(prev_unit):
                    new_progress = tmp
                    next_unit = unit
                    hopped = True
                    break
            if not hopped:
                new_progress["done"] = True
                next_unit = ""
        db_update_thread_fields.invoke({
            "user_id": user_id,
            "thread_id": thread_id,
            "fields_json": json.dumps({"speech_progress": new_progress}, ensure_ascii=False)
        })
        chk = db_get_thread.invoke({"user_id": user_id, "thread_id": thread_id}) or {}
        print(f"[CALL {_call_id}] AFTER_SAVE(correct) speech_progress={chk.get('speech_progress')}", flush=True)
        if new_progress.get("done"):
            fb = result.get("feedback") or "Tot."
            return f"{fb}\n\nBai hom nay da ket thuc. Minh dang tinh diem, ban cho minh mot chut nhe."

        fb = result.get("feedback")
        if fb:
            return f"{fb}\n\n{next_unit}"
        return next_unit

    # ===== C) Nếu chưa hỏi bước nào -> hỏi bước hiện tại =====
    review_text = ""
    out_text = _render_one_unit(plan, progress)
    new_progress = dict(progress)
    new_progress["awaiting_answer"] = True
    new_progress["attempts"] = 0
    db_update_thread_fields.invoke({
        "user_id": user_id,
        "thread_id": thread_id,
        "fields_json": json.dumps({"speech_progress": new_progress}, ensure_ascii=False)
    })
    chk = db_get_thread.invoke({"user_id": user_id, "thread_id": thread_id}) or {}
    print(f"[CALL {_call_id}] AFTER_SAVE speech_progress={chk.get('speech_progress')}")
    if review_text:
        return f"{review_text}\n\n{out_text}"
    return out_text

