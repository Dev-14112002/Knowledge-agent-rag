from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

DB_PATH = "./chroma_db"


def load_vector_store():

    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    return vector_store


def create_qa_chain():

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    return qa_chain


def create_vector_store(chunks):

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory=DB_PATH
    )

    return vector_store
