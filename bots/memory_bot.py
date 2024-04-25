from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI


def main():
    print("hello memory_bot")
    # Define an OpenAI chat model, no memory at all
    llm = ChatOpenAI(temperature=0)

    # Create a chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "Respond to question: {question}"),
        ]
    )

    # Insert a question into the template and call the model
    full_prompt = prompt_template.format_messages(
        question="What do you know about enabling memory for chatbots using langchain "
    )
    llm.invoke(full_prompt)
    print(f"full_prompt: {full_prompt}")

    full_prompt = prompt_template.format_messages(
        question="what did i ask you earlier?"
    )
    llm.invoke(full_prompt)

    chat = ChatOpenAI(temperature=0)
    history = ChatMessageHistory()

    history.add_ai_message("Hi! Ask me anything about Steve Jobs")
    history.add_user_message(
        " Describe a metaphor for Steve Jobs life and his working style"
    )
    history.add_ai_message(chat(history.messages))

    print(f"history.messages: {history.messages}")

    history.add_user_message("Summarize the previous response")
    history.add_ai_message(chat(history.messages))

    chat = OpenAI(temperature=0)
    memory = ConversationBufferMemory(size=5)  # sliding window buffer of size 5
    buffer_chain = ConversationChain(llm=chat, memory=memory, verbose=True)

    buffer_chain.predict(input="What is Apple's mission statement?")
    buffer_chain.predict(input="Summarize your last response")
    buffer_chain.predict(input="Summarize it in even lesser words")
    buffer_chain.predict(input="What was the first thing i asked you?")

    chat = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    memory = ConversationSummaryMemory(llm=chat)

    summary_chain = ConversationChain(llm=chat, memory=memory, verbose=True)

    summary_chain.predict(input="What is Apple's mission statement?")
    summary_chain.predict(input="Summarize your last response")
    summary_chain.predict(input="Summarize it in even lesser words")
    summary_chain.predict(input="What was the first thing i asked you?")


if __name__ == "__main__":
    load_dotenv()
    main()
