from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.agents import load_tools


def main():
    print("hello agents...")
    llm = OpenAI(temperature=0)

    tool_names = ["serpapi"]
    tools = load_tools(tool_names)
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )
    agent.run("What is LangChain?")
    # Input should be a search query.
    agent.run("who is the ceo of pipe?")

    tool_names = ["wolfram-alpha"]
    tools = load_tools(tool_names)
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )
    agent.run("What is Wolfram alpha?")
    # Input should be a search query.
    agent.run("What is the asthenosphere?")


if __name__ == "__main__":
    load_dotenv()
    main()
