import os

from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain.document_loaders import (
    TextLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone


def main_using_pinecone():
    print("main_using_pinecone...")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    index_name = "langchaintest"
    print(
        f"PINECONE_API_KEY is {PINECONE_API_KEY} created at {PINECONE_API_ENV} using {index_name}"
    )

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index = pc.Index(index_name)
    if index_name not in pc.list_indexes().names():
        print(f"index {index_name} is NOT present!")
    else:
        print(f"index {index_name} IS present")
    embeddings = OpenAIEmbeddings()
    loader = TextLoader(file_path="../data/PaulGrahamEssays/vb.txt")

    data = loader.load()
    print(f"You have {len(data)} document(s) in your data")
    print(f"There are {len(data[0].page_content)} characters in your sample document")
    print(f"Here is a sample: {data[0].page_content[:200]}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(data)
    print(f"Now you have {len(documents)} documents")
    # Upsert the documents into the index
    print("Uploading files to Pinecone index...")

    PineconeLangChain.from_documents(documents, embeddings, index_name=index_name)

    print("Files uploaded.")

    # Now we have the documents in the index, let's perform the search
    query = "What is great about having kids?"
    docs = index.query(queries=[embeddings.embed_query(query)], top_k=10)
    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    output = chain.run(input_documents=docs, question=query)
    print(output)


def main_using_chroma():
    print("main_using_chroma...")
    loader = TextLoader(file_path="../data/PaulGrahamEssays/vb.txt")

    ## Other options for loaders
    # loader = PyPDFLoader("../data/field-guide-to-data-science.pdf")
    # loader = UnstructuredPDFLoader("../data/field-guide-to-data-science.pdf")
    # loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")

    data = loader.load()
    # Note: If you're using PyPDFLoader then it will split by page for you already
    print(f"You have {len(data)} document(s) in your data")
    print(f"There are {len(data[0].page_content)} characters in your sample document")
    print(f"Here is a sample: {data[0].page_content[:200]}")
    # We'll split our data into chunks around 500 characters each with a 50 character overlap. These are relatively small.

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)

    # Let's see how many small chunks we have
    print(f"Now you have {len(texts)} documents")

    embeddings = OpenAIEmbeddings()
    # load it into Chroma
    vectorstore = Chroma.from_documents(texts, embeddings)
    query = "What is great about having kids?"
    docs = vectorstore.similarity_search(query)
    # Here's an example of the first document that was returned
    for doc in docs:
        print(f"{doc.page_content}\n")
    print("Done!")


if __name__ == "__main__":
    load_dotenv()
    main_using_chroma()
    # main_using_pinecone()
