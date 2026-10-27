import logging
import os

import ollama
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

DOC_PATH = "./data/OMA-TS-LightweightM2M_Core-V1_1-20180710-A.pdf"
MODEL_NAME = "lwm2m"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


def ingest_pdf(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info(f"Loaded {doc_path}")
        return data
    else:
        logging.error(f"File {doc_path} does not exist")
        return None


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Split documents into chunks")
    return chunks


def create_vector_db(chunks):
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector DB created")
    return vector_db


def create_retriever(vector_db, llm):
    QUERRY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to
        generate three different versions of the given user questin to retrieve
        relevant documents from a vector database. By generating multiple
        perspectives on the user question, your goal is to help the user
        overcome some of the limitations of the distance-based similarity
        search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERRY_PROMPT
    )
    logging.info("Retriever created")
    return retriever


def create_chain(retriever, llm):
    # RAG prompt
    template = ("""Answer the question based ONLY on the following context:
{context}
Question: {question}""")

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created successfully")
    return chain


def main():

    data = ingest_pdf(DOC_PATH)
    if data is None:
        return

    chunks = split_documents(data)

    vector_db = create_vector_db(chunks)

    ollama.create(model=MODEL_NAME,
                  from_="gemma4:e2b",
                  system="You are an expert that knows everything about LwM2M 1.1, CoAP and IoT. You provide succinct answers.",
                  parameters={"temperature": 0.1},)

    llm = ChatOllama(model=MODEL_NAME)

    retriever = create_retriever(vector_db, llm)

    chain = create_chain(retriever, llm)

    example_question = "Which transport technologies are supported?"

    # get answer
    res = chain.invoke(input=example_question)
    print(res)


if __name__ == "__main__":
    main()
