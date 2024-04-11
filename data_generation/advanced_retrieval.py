from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Set logging for the queries
import logging

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


def multi_query():
    print("multi_query...")
    path = "../data/PaulGrahamEssaysLarge/"
    loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    print(f"You have {len(docs)} essays loaded from {path}")
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    print(f"Your {len(docs)} documents have been split into {len(splits)} chunks")
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
    if (
        "vectordb" in globals()
    ):  # If you've already made your vectordb this will delete it so you start fresh
        vectordb.delete_collection()
        print("Vector Database deleted!")

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    question = "What is the authors view on the early stages of a startup?"
    llm = ChatOpenAI(temperature=0)

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )
    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    print(len(unique_docs))
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    output = llm.predict(
        text=PROMPT.format_prompt(context=unique_docs, question=question).text
    )

    print(output)


def contextual_compression():

    print("contextual_compression...")
    llm = ChatOpenAI(temperature=0)
    path = "../data/PaulGrahamEssaysLarge/"
    loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    print(f"You have {len(docs)} essays loaded from {path}")
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    print(f"Your {len(docs)} documents have been split into {len(splits)} chunks")
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
    if (
        "vectordb" in globals()
    ):  # If you've already made your vectordb this will delete it so you start fresh
        vectordb.delete_collection()
        print("Vector Database deleted!")

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=vectordb.as_retriever()
    )
    print(splits[0].page_content)
    compressor.compress_documents(
        documents=[splits[0]], query="test for what you like to do"
    )
    question = "What is the authors view on the early stages of a startup?"
    compressed_docs = compression_retriever.get_relevant_documents(question)
    print(len(compressed_docs))
    print(f"compresed_docs: {compressed_docs}")

    prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
        {context}
    
        Question: {question}
        Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    output = llm.predict(
        text=PROMPT.format_prompt(context=compressed_docs, question=question).text
    )

    print(output)


def parent_document_retriever():
    print("parent_document_retriever")
    # This text splitter is used to create the child documents. They should be small chunk size.
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="return_full_documents", embedding_function=OpenAIEmbeddings()
    )
    # The storage layer for the parent documents
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    llm = ChatOpenAI(temperature=0)
    path = "../data/PaulGrahamEssaysLarge/"
    loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    print(f"You have {len(docs)} essays loaded from {path}")
    retriever.add_documents(docs, ids=None)
    sub_docs = vectorstore.similarity_search("what is some investing advice?")
    print(sub_docs)

    retrieved_docs = retriever.get_relevant_documents("what is some investing advice?")
    print(retrieved_docs[0].page_content[:1000])

    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="return_split_parent_documents",
        embedding_function=OpenAIEmbeddings(),
    )

    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(docs)
    sub_docs = vectorstore.similarity_search("what is some investing advice?")
    print(sub_docs)

    larger_chunk_relevant_docs = retriever.get_relevant_documents(
        "what is some investing advice?"
    )
    print(larger_chunk_relevant_docs[0])

    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    question = "what is some investing advice?"

    output = llm.predict(
        text=PROMPT.format_prompt(
            context=larger_chunk_relevant_docs, question=question
        ).text
    )

    print(output)


def ensemble_retriever():
    print("ensemble_retriever...")
    llm = ChatOpenAI(temperature=0)
    path = "../data/PaulGrahamEssaysLarge/"
    loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    print(f"You have {len(docs)} essays loaded from {path}")
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    print(f"Your {len(docs)} documents have been split into {len(splits)} chunks")
    # initialize the bm25 retriever and faiss retriever
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 2
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(splits, embedding)
    vectordb = vectordb.as_retriever(search_kwargs={"k": 2})
    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectordb], weights=[0.5, 0.5]
    )
    ensemble_docs = ensemble_retriever.get_relevant_documents(
        "what is some investing advice?"
    )
    len(ensemble_docs)
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    question = "what is some investing advice?"
    print(f"question is {question}")
    output = llm.predict(
        text=PROMPT.format_prompt(context=ensemble_docs, question=question).text
    )

    print(output)


def self_querying():
    print("self_querying...")
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    path = "../data/PaulGrahamEssaysLarge/"
    loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    print(f"You have {len(docs)} essays loaded from {path}")
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    print(f"Your {len(docs)} documents have been split into {len(splits)} chunks")

    vectorstore = Chroma.from_documents(splits, embeddings)

    if (
        "vectorstore" in globals()
    ):  # If you've already made your vectordb this will delete it so you start fresh
        vectorstore.delete_collection()

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The filename of the essay",
            type="string or list[string]",
        ),
    ]

    document_content_description = "Essays from Paul Graham"
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
        enable_limit=True,
    )

    import re

    retriever.get_relevant_documents(
        "Return only 1 essay. What is one thing you can do to figure out what you like to do from source '../data/PaulGrahamEssaysLarge/island.txt'"
    )

    for split in splits:
        split.metadata["essay"] = re.search(
            r"[^/]+(?=\.\w+$)", split.metadata["source"]
        ).group()

    metadata_field_info = [
        AttributeInfo(
            name="essay",
            description="The name of the essay",
            type="string or list[string]",
        ),
    ]
    if (
        "vectorstore" in globals()
    ):  # If you've already made your vectordb this will delete it so you start fresh
        vectorstore.delete_collection()

    vectorstore = Chroma.from_documents(splits, embeddings)
    document_content_description = "Essays from Paul Graham"
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
        enable_limit=True,
    )
    query = "Tell me about investment advice the 'worked' essay? return only 1"
    output = retriever.get_relevant_documents(query)
    print(f"query is {query}")
    print(f"output is {output}")


if __name__ == "__main__":
    load_dotenv()
    multi_query()
    contextual_compression()
    parent_document_retriever()
    ensemble_retriever()
    self_querying()
