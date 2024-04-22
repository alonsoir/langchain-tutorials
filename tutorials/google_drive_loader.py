from dotenv import load_dotenv
from langchain.document_loaders import GoogleDriveLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain

def main():
    print("main...")
    loader = GoogleDriveLoader(
        document_ids=["1BT5apJMTUvG9_59-ceHbuZXVTJKeyyknQsz9ZNIEwQ8"],
        credentials_path="../../desktop_credetnaisl.json",
    )
    docs = loader.load()
    llm = OpenAI(temperature=0)

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    chain.run(docs)

    query = "What problem is this product solving?"

    chain = load_qa_chain(llm, chain_type="stuff")
    chain.run(input_documents=docs, question=query)

    loader = GoogleDriveLoader(
        document_ids=["1ETW9siA8EBkdms7R1UTw1ysXag2U6gb9wMzRpSxjqY8"],
        credentials_path="../../desktop_credetnaisl.json",
    )
    new_doc = loader.load()

    docs.extend(new_doc)

    query = "What are users saying about our product?"

    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    chain.run(input_documents=docs, question=query)

    type(docs)

    docs.pop()
    print(docs[0].dict())


if __name__ == "__main__":
    load_dotenv()
    main()
