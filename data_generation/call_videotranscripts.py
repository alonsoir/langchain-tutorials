from dotenv import load_dotenv

# To get environment variables
import os

# Make the display a bit wider
from IPython.display import display, HTML

display(HTML("<style>.container { width:90% !important; }</style>"))

# To split our transcript into pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Our chat model. We'll use the default which is gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

# Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,  # I included this one so you know you'll have it but we won't be using it
    HumanMessagePromptTemplate,
)

# To create our chat messages
from langchain.schema import AIMessage, HumanMessage, SystemMessage


def main():
    print("main...")
    with open("../data/Transcripts/acme_co_v2.txt", "r") as file:
        content = file.read()
    print("Transcript:\n")
    print(content[:215])  # Why 215? Because it cut off at a clean line
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250
    )
    texts = text_splitter.create_documents([content])

    print(f"You have {len(texts)} texts")
    print(texts[0])
    # Your api key should be an environment variable, or else put it here
    # We are using a chat model in case you wanted to use gpt4
    llm = ChatOpenAI(temperature=0)
    # verbose=True will output the prompts being sent to the
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    output = chain.run(texts)
    print(output)

    # Custom Prompts
    # I'm going to write custom prompts that give the AI more instructions on what role I want it to play
    template = """
    
    You are a helpful assistant that helps {sales_rep_name}, a sales rep at {sales_rep_company}, summarize information from a sales call.
    Your goal is to write a summary from the perspective of {sales_rep_name} that will highlight key points that will be relevant to making a sale
    Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"
    Do not repeat {sales_rep_name}'s name in your output
    
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "{text}"  # Simply just pass the text as a human message
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        messages=[system_message_prompt, human_message_prompt]
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=chat_prompt)

    # Because we aren't specifying a combine prompt the default one will be used

    output = chain.run(
        {
            "input_documents": texts,
            "sales_rep_company": "Marin Transitions Partner",
            "sales_rep_name": "Greg",
        }
    )
    print(output)

    summary_output_options = {
        "one_sentence": """
         - Only one sentence
        """,
        "bullet_points": """
         - Bullet point format
         - Separate each bullet point with a new line
         - Each bullet point should be concise
        """,
        "short": """
         - A few short sentences
         - Do not go longer than 4-5 sentences
        """,
        "long": """
         - A verbose summary
         - You may do a few paragraphs to describe the transcript if needed
        """,
    }

    template = """

    You are a helpful assistant that helps {sales_rep_name}, a sales rep at {sales_rep_company}, summarize information from a sales call.
    Your goal is to write a summary from the perspective of Greg that will highlight key points that will be relevant to making a sale
    Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"
    """
    system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

    human_template = "{text}"  # Simply just pass the text as a human message
    human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt_map = ChatPromptTemplate.from_messages(
        messages=[system_message_prompt_map, human_message_prompt_map]
    )

    template = """

    You are a helpful assistant that helps {sales_rep_name}, a sales rep at {sales_rep_company}, summarize information from a sales call.
    Your goal is to write a summary from the perspective of Greg that will highlight key points that will be relevant to making a sale
    Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"

    Respond with the following format
    {output_format}

    """
    system_message_prompt_combine = SystemMessagePromptTemplate.from_template(template)

    human_template = "{text}"  # Simply just pass the text as a human message
    human_message_prompt_combine = HumanMessagePromptTemplate.from_template(
        human_template
    )

    chat_prompt_combine = ChatPromptTemplate.from_messages(
        messages=[system_message_prompt_combine, human_message_prompt_combine]
    )

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=chat_prompt_map,
        combine_prompt=chat_prompt_combine,
        verbose=True,
    )

    user_selection = "one_sentence"

    output = chain.run(
        {
            "input_documents": texts,
            "sales_rep_company": "Marin Transitions Partner",
            "sales_rep_name": "Greg",
            "output_format": summary_output_options[user_selection],
        }
    )

    print(f"output is {output}")


if __name__ == "__main__":
    load_dotenv()
    main()
