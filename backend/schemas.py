from pydantic import BaseModel
from typing import List
from langchain_core.documents import Document

class UserQueryRequest(BaseModel):
    user_query: str

class UserQueryResponse(BaseModel):
    text_response: str
    documents: List[Document]
