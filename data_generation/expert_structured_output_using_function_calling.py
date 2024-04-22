from dotenv import load_dotenv

# LangChain Models
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Standard Helpers
import pandas as pd
import requests
import time
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Text Helpers
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# For token counting
from langchain.callbacks import get_openai_callback
from langchain.pydantic_v1 import BaseModel, Field
import enum
from langchain.chains import create_extraction_chain_pydantic
from typing import Sequence


class GoodOrBad(str, enum.Enum):
    GOOD = "Good"
    BAD = "Bad"


class Food(BaseModel):
    """Identifying information about a person's food review."""

    name: str = Field(..., description="Name of the food mentioned")
    good_or_bad: GoodOrBad = Field(
        ..., description="Whether or not the user thought the food was good or bad"
    )


class Person(BaseModel):
    """Someone who gives their review on different foods"""

    name: str = Field(..., description="Name of the person")
    foods: Sequence[Food] = Field(..., description="A food that a person mentioned")


class Query(BaseModel):
    """Extract the change a user would like to make to a financial forecast"""

    entity: str = Field(
        ..., description="Name of the category or account a person would like to change"
    )
    amount: int = Field(..., description="Amount they would like to change it by")
    year: int = Field(..., description="The year they would like the change to")


def printOutput(output):
    print(json.dumps(output, sort_keys=True, indent=3))


def pull_from_greenhouse(board_token):
    # If doing this in production, make sure you do retries and backoffs

    # Get your URL ready to accept a parameter
    url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true"

    try:
        response = requests.get(url)
    except:
        # In case it doesn't work
        print("Whoops, error")
        return

    status_code = response.status_code

    jobs = response.json()["jobs"]

    print(f"{board_token}: {status_code}, Found {len(jobs)} jobs")

    return jobs


# I parsed through an output to create the function below
def describeJob(job_description):
    print(f"Job ID: {job_description['id']}")
    print(f"Link: {job_description['absolute_url']}")
    print(
        f"Updated At: {datetime.fromisoformat(job_description['updated_at']).strftime('%B %-d, %Y')}"
    )
    print(f"Title: {job_description['title']}\n")
    print(f"Content:\n{job_description['content'][:550]}")


class Tool(BaseModel):
    """The name of a tool or company"""

    name: str = Field(..., description="Name of the food mentioned")


class Tools(BaseModel):
    """A tool, application, or other company that is listed in a job description."""

    tools: Sequence[Tool] = Field(
        ...,
        description=""" A tool or technology listed
        Examples:
        * "Experience in working with Netsuite, or Looker a plus." > NetSuite, Looker
        * "Experience with Microsoft Excel" > Microsoft Excel
    """,
    )


def main():
    print("hello from expert_structured_output_using_function_calling...")
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo-0613",  # Cheaper but less reliable
        temperature=0,
        max_tokens=2000,
    )
    functions = [
        {
            "name": "get_food_mentioned",
            "description": "Get the food that is mentioned in the review from the customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "food": {
                        "type": "string",
                        "description": "The type of food mentioned, ex: Ice cream",
                    },
                    "good_or_bad": {
                        "type": "string",
                        "description": "whether or not the user thought the food was good or bad",
                        "enum": ["good", "bad"],
                    },
                },
                "required": ["location"],
            },
        }
    ]

    output = chat(
        messages=[
            SystemMessage(content="You are an helpful AI bot"),
            HumanMessage(content="I thought the burgers were awesome"),
        ],
        functions=functions,
    )
    print(json.dumps(output.additional_kwargs, indent=4))

    # Using class
    output = chat(
        messages=[
            SystemMessage(content="You are an helpful AI bot"),
            HumanMessage(content="I thought the burgers were awesome"),
        ],
        functions=[
            {
                "name": "FoodExtractor",
                "description": (
                    "Identifying information about a person's food review."
                ),
                "parameters": Food.schema(),
            }
        ],
    )
    print(f"output: {output}")

    # Extraction
    chain = create_extraction_chain_pydantic(pydantic_schema=Food, llm=chat)

    # Run
    text = """I like burgers they are great"""
    chain.run(text)

    chat = ChatOpenAI(
        model_name="gpt-4-0613",  # Cheaper but less reliable
        temperature=0,
        max_tokens=2000,
    )

    # Extraction
    chain = create_extraction_chain_pydantic(pydantic_schema=Person, llm=chat)

    # Run
    text = """amy likes burgers and fries but doesn't like salads"""
    output = chain.run(text)
    print(f"text: {text}")
    print(f"output[0]: {output[0]}")

    chain = create_extraction_chain_pydantic(pydantic_schema=Query, llm=chat)
    query = "Can you please add 10 more units to inventory in 2022?"
    output = chain.run(query)
    print(f"text: {text}")

    print(f"output: {output}")

    query = "Remove 3 million from revenue in 2021"
    output = chain.run(query)
    print(f"text: {text}")

    print(f"output[0]: {output[0]}")

    jobs = pull_from_greenhouse("okta")
    # Keep in mind that my job_ids will likely change when you run this depending on the postings of the company
    job_index = 0

    print("Preview:\n")
    print(json.dumps(jobs[job_index])[:400])

    # Note: I'm using a hard coded job id below. You'll need to switch this if this job ever changes
    # and it most definitely will!
    job_id = 5856411

    job_description = [item for item in jobs if item["id"] == job_id][0]

    describeJob(job_description)

    soup = BeautifulSoup(job_description["content"], "html.parser")

    text = soup.get_text()

    # Convert your html to markdown. This reduces tokens and noise
    text = md(text)

    print(f"text: {text[:600]}")

    # chain = create_extraction_chain_pydantic(pydantic_schema=Tools, llm=chat)
    # chain espera un Diccionario!
    # import spacy
    from spacy.matcher import Matcher

    # Cargar el modelo de procesamiento de texto
    # nlp = spacy.load('en_core_web_lg')

    # Procesar el texto
    # doc = nlp(text)

    # Crear un diccionario vacío
    # data = {}
    # Agregar secciones al diccionario
    # section_name = None
    # for ent in doc.ents:
    #    print(f"ent is {ent}")
    #    section_name = ent.label_
    #    if section_name not in data:
    #        data[section_name] = []
    #    data[section_name].append(ent.text)

    # Agregar valores adicionales al diccionario de datos
    # data['otra_sección'] = ['valor1', 'valor2', 'valor3']

    # Imprimir el diccionario resultante
    # print(f"data is {data}")

    # output = chain(data)
    # print(f"output: {output['text']}")

    # with get_openai_callback() as cb:
    #    result = chain(output)
    #    print(f"Total Tokens: {cb.total_tokens}")
    #    print(f"Prompt Tokens: {cb.prompt_tokens}")
    #    print(f"Completion Tokens: {cb.completion_tokens}")
    #    print(f"Successful Requests: {cb.successful_requests}")
    #    print(f"Total Cost (USD): ${cb.total_cost}")

    # print(f"text is {text}")
    # print(f"result is :{result}")
    print("Done!")


if __name__ == "__main__":
    load_dotenv()
    main()
