from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import os


def generate_response(llm, mentioned_parent_tweet_text):
    # It would be nice to bring in information about the links, pictures, etc.
    # But out of scope for now
    system_template = """
        You are an incredibly wise and smart tech mad scientist from silicon valley.
        Your goal is to give a concise prediction in response to a piece of text from the user.

        % RESPONSE TONE:

        - Your prediction should be given in an active voice and be opinionated
        - Your tone should be serious w/ a hint of wit and sarcasm

        % RESPONSE FORMAT:

        - Respond in under 200 characters
        - Respond in two or less short sentences
        - Do not respond with emojis

        % RESPONSE CONTENT:

        - Include specific examples of old tech if they are relevant
        - If you don't have an answer, say, "Sorry, my magic 8 ball isn't working right now 🔮"
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    final_prompt = chat_prompt.format_prompt(
        text=mentioned_parent_tweet_text
    ).to_messages()
    response = llm(final_prompt).content

    return response


if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(
        temperature=0.3,
        model_name="gpt-3.5-turbo",
    )
    tweet = """
    I wanted to build a sassy Twitter Bot that responded about the 'good ole days' of tech

    @SiliconOracle was built using @LangChainAI and hosted on @railway 

    Condensed Prompt:
    You are a mad scientist from old school silicon valley that makes predictions on the future of a tweet
    """

    response = generate_response(llm, tweet)
    print(response)
