from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage


def main():
    print("hello chatapi...")
    chat = ChatOpenAI()
    query = "What is the name of the most populous state in the USA?"
    message = [HumanMessage(content=query)]
    resp = chat(message)
    print(f"query is {query}")
    print(f"resp is {resp.content}")
    querySystemMessage = "Say the opposite of what the user says"
    queryHumanMessage = "I love programming."
    messages = [
        SystemMessage(content=querySystemMessage),
        HumanMessage(content=queryHumanMessage),
    ]
    resp = chat(messages)
    print(f"query is {querySystemMessage}")
    print(f"query is {queryHumanMessage}")
    print(f"resp is {resp.content}")

    messages = [
        SystemMessage(content="Say the opposite of what the user says"),
        HumanMessage(content="I love programming."),
        AIMessage(content="I hate programming."),
        HumanMessage(content="The moon is out"),
    ]
    print(chat(messages).content)
    messages = [
        SystemMessage(content="Say the opposite of what the user says"),
        HumanMessage(content="I love programming."),
        AIMessage(content="I hate programming."),
        HumanMessage(content="What is the first thing that I said?"),
    ]
    print(chat(messages).content)
    batch_messages = [
        [
            SystemMessage(
                content="You are a helpful word machine that creates an alliteration using a base word"
            ),
            HumanMessage(content="Base word: Apple"),
        ],
        [
            SystemMessage(
                content="You are a helpful word machine that creates an alliteration using a base word"
            ),
            HumanMessage(content="Base word: Dog"),
        ],
    ]
    chat.generate(batch_messages)
    # Make SystemMessagePromptTemplate
    prompt = PromptTemplate(
        template="Propose creative ways to incorporate {food_1} and {food_2} in the cuisine of the users choice.",
        input_variables=["food_1", "food_2"],
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    # Output of system_message_prompt
    system_message_prompt.format(food_1="Bacon", food_2="Shrimp")
    # Make HumanMessagePromptTemplate
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    # Create ChatPromptTemplate: Combine System + Human
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chat_prompt_with_values = chat_prompt.format_prompt(
        food_1="Bacon", food_2="Shrimp", text="I really like food from Germany."
    )

    chat_prompt_with_values.to_messages()
    response = chat(chat_prompt_with_values.to_messages()).content
    print(f"response is: {response}\n\n")

    chat = ChatOpenAI(
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
        temperature=0,
    )

    resp = chat(chat_prompt_with_values.to_messages())
    print(f"\nresp is: {resp.content}")


if __name__ == "__main__":
    load_dotenv()
    main()
