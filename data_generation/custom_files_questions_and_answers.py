import nltk
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def main():
    print("hello custom_files_questions_and_answers")
    nltk.download("averaged_perceptron_tagger")
    # Get your loader ready
    loader = DirectoryLoader("../data/PaulGrahamEssaySmall/", glob="**/*.txt")
    # Load up your text into documents
    documents = loader.load()
    # Get your text splitter ready
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Split your documents into texts
    texts = text_splitter.split_documents(documents)

    # Turn your texts into embeddings
    embeddings = OpenAIEmbeddings()

    # Get your docsearch ready
    docsearch = FAISS.from_documents(texts, embeddings)
    # Load up your LLM
    llm = OpenAI()

    # Create your Retriever
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )

    # Run a query
    query = "What did McCarthy discover?"
    print(f"query: {query}")
    output = qa.run(query)
    print(f"output: {output}")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    query = "What did McCarthy discover?"
    print(f"query: {query}")
    result = qa({"query": query})
    print(f"resultado: {result['result']}")
    print(f"source_documents: {result['source_documents']}")


if __name__ == "__main__":
    load_dotenv()
    main()
