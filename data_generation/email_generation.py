from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import pandas as pd


def get_company_page(company_path):
    y_combinator_url = f"https://www.ycombinator.com{company_path}"

    print(y_combinator_url)

    loader = UnstructuredURLLoader(urls=[y_combinator_url])
    return loader.load()


def main():
    print("main...")
    data = get_company_page("/companies/poly")
    # Get the data of the company you're interested in

    print(f"You have {len(data)} document(s)")
    print(f"Preview of your data:\n\n{data[0].page_content[:30]}")
    # Split up the texts so you don't run into token limits
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=800,
        chunk_overlap=0,
    )
    docs = text_splitter.split_documents(data)
    print(f"You now have {len(docs)} documents")

    map_prompt = """Below is a section of a website about {prospect}

    Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.

    {text}

    % CONCISE SUMMARY:"""
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "prospect"]
    )
    combine_prompt = """
    Your goal is to write a personalized outbound email from {sales_rep}, a sales rep at {company} to {prospect}.

    A good email is personalized and combines information about the two companies on how they can help each other.
    Be sure to use value selling: A sales methodology that focuses on how your product or service will provide value to the customer instead of focusing on price or solution.

    % INFORMATION ABOUT {company}:
    {company_information}

    % INFORMATION ABOUT {prospect}:
    {text}

    % INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:
    - Start the email with the sentence: "We love that {prospect} helps teams..." then insert what they help teams do.
    - The sentence: "We can help you do XYZ by ABC" Replace XYZ with what {prospect} does and ABC with what {company} does 
    - A 1-2 sentence description about {company}, be brief
    - End your email with a call-to-action such as asking them to set up time to talk more

    % YOUR RESPONSE:
    """
    combine_prompt_template = PromptTemplate(
        template=combine_prompt,
        input_variables=[
            "sales_rep",
            "company",
            "prospect",
            "text",
            "company_information",
        ],
    )
    company_information = """
    * RapidRoad helps product teams build product faster
    * We have a platform that allows product teams to talk more, exchange ideas, and listen to more customers
    * Automated project tracking: RapidRoad could use machine learning algorithms to automatically track project progress, identify potential bottlenecks, and suggest ways to optimize workflows. This could help product teams stay on track and deliver faster results.
    * Collaboration tools: RapidRoad could offer built-in collaboration tools, such as shared task lists, real-time messaging, and team calendars. This would make it easier for teams to communicate and work together, even if they are in different locations or time zones.
    * Agile methodology support: RapidRoad could be specifically designed to support agile development methodologies, such as Scrum or Kanban. This could include features like sprint planning, backlog management, and burndown charts, which would help teams stay organized and focused on their goals.
    """

    llm = OpenAI(temperature=0.7)

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=True,
    )
    output = chain(
        {
            "input_documents": docs,  # The seven docs that were created before
            "company": "RapidRoad",
            "company_information": company_information,
            "sales_rep": "Greg",
            "prospect": "Poly",
        }
    )
    print(output["output_text"])

    df_companies = pd.read_clipboard()
    print(df_companies)
    """
    for i, company in df_companies.iterrows():
        print(f"{i + 1}. {company['Name']}")
        page_data = get_company_page(company["Link"])
        docs = text_splitter.split_documents(page_data)

        output = chain(
            {
                "input_documents": docs,
                "company": "RapidRoad",
                "sales_rep": "Greg",
                "prospect": company["Name"],
                "company_information": company_information,
            }
        )

        print(output["output_text"])
        print("\n\n")
    """
    print("Done")


if __name__ == "__main__":
    load_dotenv()
    main()
