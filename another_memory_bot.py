from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory

from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI


def conversationBufferMemory(llm):
    print("conversationBufferMemory...")
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    conversation.predict(input="Hi, my name is Alonso")
    conversation.predict(input="What is 1+1?")
    conversation.predict(
        input="What is the weather like today using Celsius, where am i located?,"
        " provide more specific details about the weather forecast for my actual location."
    )
    conversation.predict(input="What is my name?")
    print(f"memory.buffer: {memory.buffer}")
    memory.load_memory_variables({})

    memory = ConversationBufferMemory()
    memory.save_context({"input": "Hi"}, {"output": "What's up"})
    print(memory.buffer)


def conversationBufferWindowMemory(llm):
    print("conversationBufferWindowMemory...")
    memory = ConversationBufferWindowMemory(k=1)
    memory.load_memory_variables({})
    memory.save_context({"input": "Hi"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    memory.load_memory_variables({})
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    conversation.predict(input="Hi, my name is Alonso")
    conversation.predict(input="What is 1+1?")
    conversation.predict(input="What is my name?")
    conversation.predict(
        input="What is the weather like today using Celsius,"
        " i am currently living at Badajoz, Extremadura, Spain. Provide more specific details about the weather forecast for my actual location."
    )


def conversationTokenBufferMemory(llm):
    print("conversationTokenBufferMemory...")
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
    memory.save_context({"input": "AI is what?!"}, {"output": "Amazing!"})
    memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful!"})
    memory.save_context({"input": "Chatbots are what?"}, {"output": "Charming!"})

    memory.load_memory_variables({})
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    conversation.predict(input="Hi, what is the AI?")
    conversation.predict(input="What is the backpropagation?")
    conversation.predict(input="What are chatbots?")
    print(f"memory.buffer: {memory.buffer}")


def conversation_summary_memory(llm):
    print("conversation_summary_memory...")
    # create a long string
    schedule = "There is a meeting at 8am with your product team. \
    You will need your powerpoint presentation prepared. \
    9am-12pm have time to work on your LangChain \
    project which will go quickly because Langchain is such a powerful tool. \
    At Noon, lunch at the italian resturant with a customer who is driving \
    from over an hour away to meet you to understand the latest in AI. \
    Be sure to bring your laptop to show the latest LLM demo."

    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    memory.save_context(
        {"input": "What is on the schedule today?"}, {"output": f"{schedule}"}
    )

    memory.load_memory_variables({})

    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=400)
    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    memory.save_context(
        {"input": "What is on the schedule today?"}, {"output": f"{schedule}"}
    )

    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    conversation.predict(input="What would be a good demo to show?")


if __name__ == "__main__":
    load_dotenv()
    llm_model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    conversationBufferMemory(llm)
    conversationBufferWindowMemory(llm)
    conversationTokenBufferMemory(llm)
    conversation_summary_memory(llm)
