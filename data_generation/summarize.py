from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Taking out the warnings
import warnings
from warnings import simplefilter

from langchain.agents import initialize_agent, Tool
from langchain.utilities import WikipediaAPIWrapper


def level1():
    print("hello level1!")
    llm = OpenAI(temperature=0)
    prompt = """
    Please provide a summary of the following text

    TEXT:
    Philosophy (from Greek: φιλοσοφία, philosophia, 'love of wisdom') \
    is the systematized study of general and fundamental questions, \
    such as those about existence, reason, knowledge, values, mind, and language. \
    Some sources claim the term was coined by Pythagoras (c. 570 – c. 495 BCE), \
    although this theory is disputed by some. Philosophical methods include questioning, \
    critical discussion, rational argument, and systematic presentation.
    """
    num_tokens = llm.get_num_tokens(prompt)
    print(f"Our first prompt has {num_tokens} tokens")
    output = llm(prompt)
    print(output)
    # better prompt
    prompt = """
    Please provide a summary of the following text.
    Please provide your output in a manner that a 5 year old would understand

    TEXT:
    Philosophy (from Greek: φιλοσοφία, philosophia, 'love of wisdom') \
    is the systematized study of general and fundamental questions, \
    such as those about existence, reason, knowledge, values, mind, and language. \
    Some sources claim the term was coined by Pythagoras (c. 570 – c. 495 BCE), \
    although this theory is disputed by some. Philosophical methods include questioning, \
    critical discussion, rational argument, and systematic presentation.
    """

    num_tokens = llm.get_num_tokens(prompt)
    print(f"Our second prompt has {num_tokens} tokens")

    output = llm(prompt)
    print(output)


def level2():

    print("level2...")
    llm = OpenAI(temperature=0)

    paul_graham_essays = [
        "../data/PaulGrahamEssaySmall/getideas.txt",
        "../data/PaulGrahamEssaySmall/noob.txt",
    ]

    essays = []

    for file_name in paul_graham_essays:
        with open(file_name, "r") as file:
            essays.append(file.read())

    for i, essay in enumerate(essays):
        print(f"Essay #{i + 1}: {essay[:300]}\n")

    template = """
    Please write a one sentence summary of the following text:

    {essay}
    """

    prompt = PromptTemplate(input_variables=["essay"], template=template)

    for essay in essays:
        summary_prompt = prompt.format(essay=essay)

        num_tokens = llm.get_num_tokens(summary_prompt)
        print(f"This prompt + essay has {num_tokens} tokens")

        summary = llm(summary_prompt)

        print(f"Summary: {summary.strip()}")
        print("\n")


def level3():
    print("level3...")
    llm = OpenAI(temperature=0)
    paul_graham_essay = "../data/PaulGrahamEssays/startupideas.txt"

    with open(paul_graham_essay, "r") as file:
        essay = file.read()

    llm.get_num_tokens(essay)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )

    docs = text_splitter.create_documents([essay])

    num_docs = len(docs)

    num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

    print(
        f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens"
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=True,  # Set verbose=True if you want to see the prompts being used
    )
    output = summary_chain.run(docs)
    print(output)

    map_prompt = """
    Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    Write a concise summary of the following text delimited by triple backquotes.
    Return your response in bullet points which covers the key points of the text.
    ```{text}```
    BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text"]
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=True,
    )
    output = summary_chain.run(docs)
    print(output)


def level4():
    print("level4...")
    llm = OpenAI(temperature=0)
    # Load the book
    loader = PyPDFLoader("../data/IntoThinAirBook.pdf")
    pages = loader.load()

    # Cut out the open and closing parts
    pages = pages[26:277]

    # Combine the pages, and replace the tabs with spaces
    text = ""

    for page in pages:
        text += page.page_content

    text = text.replace("\t", " ")
    num_tokens = llm.get_num_tokens(text)

    print(f"This book has {num_tokens} tokens in it")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000
    )

    docs = text_splitter.create_documents([text])
    num_documents = len(docs)

    print(f"Now our book is split up into {num_documents} documents")
    embeddings = OpenAIEmbeddings()

    vectors = embeddings.embed_documents([x.page_content for x in docs])
    # Assuming 'embeddings' is a list or array of 1536-dimensional embeddings

    # Choose the number of clusters, this can be adjusted based on the book's content.
    # I played around and found ~10 was the best.
    # Usually if you have 10 passages from a book you can tell what it's about
    num_clusters = 11

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    # print(kmeans.labels_)
    # Filter out FutureWarnings
    simplefilter(action="ignore", category=FutureWarning)

    # Perform t-SNE and reduce to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    vectors_array = np.array(vectors)
    reduced_data_tsne = tsne.fit_transform(vectors_array)

    # Plot the reduced data
    plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=kmeans.labels_)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Book Embeddings Clustered")
    plt.show()

    # Find the closest embeddings to the centroids

    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    print(selected_indices)

    llm3 = ChatOpenAI(temperature=0, max_tokens=1000, model="gpt-3.5-turbo")
    map_prompt = """
    You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    Your response should be at least three paragraphs and fully encompass what was said in the passage.

    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(
        llm=llm3, chain_type="stuff", prompt=map_prompt_template
    )
    selected_docs = [docs[doc] for doc in selected_indices]
    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])

        # Append that summary to your list
        summary_list.append(chunk_summary)

        print(
            f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n"
        )

    summaries = "\n".join(summary_list)

    # Convert it back to a document
    summaries = Document(page_content=summaries)

    print(f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")

    llm4 = ChatOpenAI(
        temperature=0, max_tokens=3000, model="gpt-4", request_timeout=120
    )

    combine_prompt = """
    You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
    Your goal is to give a verbose summary of what happened in the story.
    The reader should be able to grasp what happened in the book.

    ```{text}```
    VERBOSE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text"]
    )
    reduce_chain = load_summarize_chain(
        llm=llm4,
        chain_type="stuff",
        prompt=combine_prompt_template,
        verbose=True,  # Set this to true if you want to see the inner workings
    )
    output = reduce_chain.run([summaries])
    print(output)


def level5():
    print("level5...")
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    wikipedia = WikipediaAPIWrapper()
    tools = [
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Useful for when you need to get information from wikipedia about a single topic",
        ),
    ]
    # AgentType.ZERO_SHOT_REACT_DESCRIPTION
    agent_executor = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )
    output = agent_executor.run(
        "Can you please provide a quick summary of Napoleon Bonaparte? \
                              Then do a separate search and tell me what the commonalities are with Serena Williams"
    )
    print(output)


if __name__ == "__main__":
    load_dotenv()
    level1()
    level2()
    level3()
    level4()
    level5()
