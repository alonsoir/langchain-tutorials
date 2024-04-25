from getpass import getpass

from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import OpenAI


def google():
    api_key = getpass()
    print(f"hello google. {api_key}")

    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)
    print(
        llm.invoke(
            "What are some of the pros and cons of Python as a programming language?"
        )
    )
def main():
    print("hello agents. ")
    llm_openAI = OpenAI(temperature=0)

    tool_names = [
        "serpapi",
        "wolfram-alpha",
        # "google-cloud-vision",
        # "ibm-watson",
        # "azure-text-analytics",
        # "amazon-textract",
        # "stanford-corenlp",
        # "open-meteo-api",
        # "google-translate",
        # "openweathermap",
        # "spotify",
        # "twilio",
        # "zillow"
    ]
    tools = load_tools(tool_names)
    agent_openAI = initialize_agent(
        tools,
        llm_openAI,
        agent="zero-shot-react-description",
        verbose=True
    )
    agent_openAI.run("What is LangChain?")
    agent_openAI.run("who is the ceo of pipe?")
    agent_openAI.run("What is Wolfram alpha?")
    agent_openAI.run("What is the asthenosphere?")
    agent_openAI.run("cual es la raiz cuadrada de 144?")



if __name__ == "__main__":
    load_dotenv()
    main()
    # google()