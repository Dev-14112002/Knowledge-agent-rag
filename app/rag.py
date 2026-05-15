from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

embedding_model = OpenAIEmbeddings()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

DB_PATH = "./chroma_db"


def load_vector_store():

    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    return vector_store


custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI research assistant.

Use the provided context and conversation history to answer the user's question.

If the question is a follow-up question, interpret it using previous conversation context.

IMPORTANT FORMATTING RULES:
- Use markdown formatting.
- Use bullet points when explaining concepts.
- Add spacing between sections.
- Keep answers clean and readable.
- Use short paragraphs.
- Highlight important terms using bold text.
- Format the response using proper markdown with headings, bullet points, and spacing.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
)


def create_qa_chain():

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )

    return qa_chain


def create_vector_store(chunks):

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory=DB_PATH
    )

    return vector_store
