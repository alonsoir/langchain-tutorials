from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd
import json
from io import StringIO


def main():
    print("main...")
    # Temp = 0 so that we get clean information without a lot of creativity
    chat_model = ChatOpenAI(temperature=0, max_tokens=1000)
    # How you would like your response structured. This is basically a fancy prompt template
    response_schemas = [
        ResponseSchema(
            name="input_industry",
            description="This is the input_industry from the user",
        ),
        ResponseSchema(
            name="standardized_industry",
            description="This is the industry you feel is most closely matched to the users input",
        ),
        ResponseSchema(
            name="match_score",
            description="A score 0-100 of how close you think the match is between user input and your match",
        ),
    ]

    # How you would like to parse your output
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # See the prompt template you created for formatting
    format_instructions = output_parser.get_format_instructions()
    # print(output_parser.get_format_instructions())

    template = """
    You will be given a series of industry names from a user.
    Find the best corresponding match on the list of standardized names.
    The closest match will be the one with the closest semantic meaning. Not just string similarity.

    {format_instructions}

    Wrap your final output with closed and open brackets (a list of json objects)

    input_industry INPUT:
    {user_industries}

    STANDARDIZED INDUSTRIES:
    {standardized_industries}

    YOUR RESPONSE:
    """

    prompt = ChatPromptTemplate(
        messages=[HumanMessagePromptTemplate.from_template(template)],
        input_variables=["user_industries", "standardized_industries"],
        partial_variables={"format_instructions": format_instructions},
    )
    # Get your standardized names. You can swap this out with whatever list you want!
    df = pd.read_csv("../data/LinkedInIndustries.csv")
    standardized_industries = ", ".join(df["Industry"].values)
    # print(standardized_industries)

    # Your user input

    user_input = "air LineZ, airline, aviation, planes that fly, farming, bread, wifi networks, twitter media agency"

    _input = prompt.format_prompt(
        user_industries=user_input, standardized_industries=standardized_industries
    )

    # print(f"There are {len(_input.messages)} message(s)")
    # print(f"Type: {type(_input.messages[0])}")
    # print("---------------------------")
    # print(_input.messages[0].content)

    output = chat_model(_input.to_messages())
    # print(type(output))
    # print(output.content)

    # Verifica si el string contiene "```json"
    if "```json" in output.content:
        # Encuentra la posición del primer "```json"
        start_index = output.content.find("```json") + len("```json")

        # Encuentra la posición del último "```"
        end_index = output.content.rfind("```")

        # Extrae el JSON entre las posiciones encontradas
        json_string = output.content[start_index:end_index].strip()
    else:
        json_string = output.content

    # print(f"json_string is {json_string}")
    # Carga el JSON como un diccionario de Python
    data = json.loads(json_string)

    # Convierte el valor de "match_score" a un número entero, no debería hacer falta
    # for item in data:
    #     item["match_score"] = int(item["match_score"])

    # Crea un objeto StringIO a partir del diccionario convertido
    json_stringio = StringIO(json.dumps(data))

    # Lee el JSON desde el objeto StringIO y crea un DataFrame
    df = pd.read_json(json_stringio)

    # Imprime el DataFrame
    print(df)
    print("Done!")


if __name__ == "__main__":
    load_dotenv()
    main()
