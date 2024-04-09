from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
import os


def main():
    print("hello ZapierToolkit")
    llm = OpenAI(temperature=0)
    zapier = ZapierNLAWrapper()
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    agent = initialize_agent(
        toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True
    )
    for tool in toolkit.get_tools():
        print(tool.name)
        print(tool.description)
        print("\n\n")
    agent.run(
        """Summarize the last email I received from greg at Data Independent.
                    Send the summary to the trending domains channel in slack."""
    )
    agent.run(
        "Get the last email I received from greg at Data Independent. Summarize the reply and create a tweet"
    )
    agent.run(
        """Get the last email I received from greg at Data Independent.
                  Create a draft email in gmail back to Greg with a good positive reply"""
    )
    agent.run(
        """Get the last email I received from greg@DataIndependent.com
                  Find a good gif that matches the intent of the email and send the gif to trending domains in slack"""
    )
    agent.run(
        """Create a tweet that says, 'langchain + zapier is great'. \
    Draft an email in gmail to greg @ data independent sharing my tweet with a personalized message"""
    )


if __name__ == "__main__":
    load_dotenv()
    main()
