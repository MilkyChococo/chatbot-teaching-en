# tools/memory_tools.py
from typing import Optional, Dict, Any
from langchain_core.tools import tool
import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from tools.schema import UpdateThreadFieldsInput
memory_store = None
def _simple_concat_summary(chat_history) -> str:
    parts = []
    for m in chat_history[-12:]:
        role = "User" if m.__class__.__name__ == "HumanMessage" else "Assistant"
        txt = (m.content or "").strip()
        if txt:
            parts.append(f"{role}: {txt}")
    return "\n".join(parts)

def init_memory_tools(store):
    global memory_store
    memory_store = store

@tool
def db_get_user_profile(user_id: str) -> Dict[str, Any]:
    """Load user profile from DB."""
    return memory_store.load_user_profile(user_id)

@tool
def db_upsert_user_profile(
    user_id: str,
    level: Optional[str] = None,
    focus: Optional[str] = None,
    session_minutes: Optional[int] = None,
    accessibility: Optional[str] = None,
) -> Dict[str, Any]:
    """Upsert user learning profile into DB, return updated profile."""
    memory_store.upsert_user_profile(
        user_id,
        level=level,
        focus=focus,
        session_minutes=session_minutes,
        accessibility=accessibility,
    )
    return memory_store.load_user_profile(user_id)

@tool
def db_get_thread(user_id: str, thread_id: str) -> Dict[str, Any]:
    """Load thread state from DB."""
    return memory_store.load_thread(user_id, thread_id)

@tool(args_schema=UpdateThreadFieldsInput)
def db_update_thread_fields(user_id: str, thread_id: str, fields_json: str):
    """Update thread document fields in MongoDB using $set. fields_json must be a JSON-encoded string."""
    fields = json.loads(fields_json) if fields_json else {}
    memory_store.update_thread_fields(user_id, thread_id, fields)
    return {"ok": True, "updated_keys": list(fields.keys())}
