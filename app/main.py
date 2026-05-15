from fastapi import FastAPI, UploadFile, File
import shutil
import os

from app.ingest import load_and_split_pdf
from app.rag import create_vector_store
from pydantic import BaseModel
from typing import List
from app.rag import create_qa_chain

app = FastAPI()


UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):

    question: str

    chat_history: List[list] = []


@app.get("/")
def home():
    return {"message": "AI Research Assistant Running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    try:

        os.makedirs("uploads", exist_ok=True)
        os.makedirs("chroma_db", exist_ok=True)

        file_path = f"{UPLOAD_DIR}/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = load_and_split_pdf(file_path)

        create_vector_store(chunks)

        return {
            "message": f"{file.filename} uploaded successfully",
            "chunks_created": len(chunks),
        }

    except Exception as e:

        return {"error": str(e)}


@app.post("/query")
def query_docs(request: QueryRequest):

    qa_chain = create_qa_chain()

    formatted_history = [tuple(chat) for chat in request.chat_history]

    response = qa_chain.invoke(
        {"question": request.question, "chat_history": formatted_history}
    )

    sources = []

    for doc in response["source_documents"]:

        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:300],
            }
        )

    return {"answer": response["answer"], "sources": sources}
