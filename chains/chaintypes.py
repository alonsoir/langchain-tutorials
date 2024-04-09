from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    print("hello chain types")
    sm_loader = UnstructuredFileLoader("../data/muir_lake_tahoe_in_winter.txt")
    sm_doc = sm_loader.load()

    lg_loader = UnstructuredFileLoader("../data/PaulGrahamEssays/worked.txt")
    lg_doc = lg_loader.load()
    doc_summary(sm_doc)
    doc_summary(lg_doc)
    llm = OpenAI()
    chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
    chain.run(sm_doc)
    chain.run(lg_doc)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    chain.run(sm_doc)
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=400,
        chunk_overlap=0,
    )
    lg_docs = text_splitter.split_documents(lg_doc)
    doc_summary(lg_docs)
    chain.run(lg_docs[:5])

    chain = load_summarize_chain(llm, chain_type="refine", verbose=True)
    chain.run(lg_docs[:5])

    chain = load_qa_chain(
        llm, chain_type="map_rerank", verbose=True, return_intermediate_steps=True
    )
    query = "Who was the authors friend who he got permission from to use the IBM 1401?"

    result = chain(
        {"input_documents": lg_docs[:5], "question": query}, return_only_outputs=True
    )
    print(result["output_text"])
    print(result["intermediate_steps"])


def doc_summary(docs):
    print(f"You have {len(docs)} document(s)")

    num_words = sum([len(doc.page_content.split(" ")) for doc in docs])

    print(f"You have roughly {num_words} words in your docs")
    print()
    print(f'Preview: \n{docs[0].page_content.split(". ")[0]}')


if __name__ == "__main__":
    load_dotenv()
    main()
