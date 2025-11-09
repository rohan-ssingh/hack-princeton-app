from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
embeddings = OpenAIEmbeddings()
image_parser = LLMImageBlobParser(model=llm)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
