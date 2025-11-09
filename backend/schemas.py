from datetime import date, datetime, time
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class ChatMessagePayload(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = Field(default="")


class UserQueryRequest(BaseModel):
    user_query: str
    conversation: List[ChatMessagePayload] = Field(default_factory=list)


class DocumentPayload(BaseModel):
    id: Optional[str] = None
    page_content: str
    metadata: Dict[str, Any]

    @staticmethod
    def _to_json_safe(value: Any) -> Any:
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, list):
            return [DocumentPayload._to_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DocumentPayload._to_json_safe(item) for item in value)
        if isinstance(value, dict):
            return {
                str(key): DocumentPayload._to_json_safe(val)
                for key, val in value.items()
            }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    def from_document(cls, document: Document) -> "DocumentPayload":
        metadata: Dict[str, Any] = {}

        for key, val in getattr(document, "metadata", {}).items():
            metadata[str(key)] = cls._to_json_safe(val)

        if "url" not in metadata:
            candidate = metadata.get("source_url") or metadata.get("source")
            if isinstance(candidate, str) and candidate:
                metadata["url"] = candidate
        if "url" not in metadata:
            candidate = metadata.get("file_name") or metadata.get("fileName")
            if isinstance(candidate, str) and candidate.startswith("http"):
                metadata["url"] = candidate

        return cls(
            id=str(getattr(document, "id", "")) or None,
            page_content=getattr(document, "page_content", ""),
            metadata=metadata,
        )


class UserQueryResponse(BaseModel):
    text_response: str
    documents: List[DocumentPayload] = Field(default_factory=list)
