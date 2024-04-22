import enum

from dotenv import load_dotenv

# Kor!
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# LangChain Models
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Standard Helpers
import pandas as pd
import requests
import time
import json
from datetime import datetime

# Text Helpers
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# For token counting
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from kor import create_extraction_chain, Object, Text, from_pydantic
from langchain.chat_models import ChatOpenAI
from kor import create_extraction_chain, Object, Text
from pydantic import Field, BaseModel
from pydantic.config import Optional
from pydantic.decorator import List


class MusicRequest(BaseModel):
    song: Optional[List[str]] = Field(
        default=None, description="The song(s) that the user would like to be played."
    )


class Action(enum.Enum):
    play = "play"
    stop = "stop"
    previous = "previous"
    next_ = "next"


class MusicRequest(BaseModel):
    song: Optional[List[str]] = Field(
        default=None, description="The song(s) that the user would like to be played."
    )
    album: Optional[List[str]] = Field(
        default=None, description="The album(s) that the user would like to be played."
    )
    artist: Optional[List[str]] = Field(
        default=None,
        description="The artist(s) whose music the user would like to hear.",
        examples=[("Songs by paul simon", "paul simon")],
    )
    action: Optional[Action] = Field(
        default=None,
        description="The action that should be taken; one of `play`, `stop`, `next`, `previous`",
        examples=[
            ("Please stop the music", "stop"),
            ("play something", "play"),
            ("play a song", "play"),
            ("next song", "next"),
        ],
    )


person_schema = Object(
    # This what will appear in your output. It's what the fields below will be nested under.
    # It should be the parent of the fields below. Usually it's singular (not plural)
    id="person",
    # Natural language description about your object
    description="Personal information about a person",
    # Fields you'd like to capture from a piece of text about your object.
    attributes=[
        Text(
            id="first_name",
            description="The first name of a person.",
        )
    ],
    # Examples help go a long way with telling the LLM what you need
    examples=[
        ("Alice and Bob are friends", [{"first_name": "Alice"}, {"first_name": "Bob"}])
    ],
)


def printOutput(output):
    print(json.dumps(output, sort_keys=True, indent=3))


def another_sample():
    llm = ChatOpenAI(
        #     model_name="gpt-3.5-turbo", # Cheaper but less reliable
        model_name="gpt-4",
        temperature=0,
        max_tokens=2000,
    )
    chain = create_extraction_chain(llm, person_schema)

    text = """
        My name is Bobby.
        My sister's name is Rachel.
        My brother's name Joe. My dog's name is Spot
    """
    output = chain.predict_and_parse(text=text)["data"]

    printOutput(output)
    # Notice how there isn't "spot" in the results list because it's the name of a dog, not a person.
    output = chain.predict_and_parse(text="The dog went to the park")["data"]
    printOutput(output)

    plant_schema = Object(
        id="plant",
        description="Information about a plant",
        # Notice I put multiple fields to pull out different attributes
        attributes=[
            Text(id="plant_type", description="The common name of the plant."),
            Text(id="color", description="The color of the plant"),
            Number(id="rating", description="The rating of the plant."),
        ],
        examples=[
            (
                "Roses are red, lilies are white and a 8 out of 10.",
                [
                    {"plant_type": "Roses", "color": "red"},
                    {"plant_type": "Lily", "color": "white", "rating": 8},
                ],
            )
        ],
    )
    text = "Palm trees are brown with a 6 rating. Sequoia trees are green"

    chain = create_extraction_chain(llm, plant_schema)
    output = chain.predict_and_parse(text=text)["data"]

    printOutput(output)

    parts = Object(
        id="parts",
        description="A single part of a car",
        attributes=[Text(id="part", description="The name of the part")],
        examples=[
            (
                "the jeep has wheels and windows",
                [{"part": "wheel"}, {"part": "window"}],
            )
        ],
    )

    cars_schema = Object(
        id="car",
        description="Information about a car",
        examples=[
            (
                "the bmw is red and has an engine and steering wheel",
                [
                    {
                        "type": "BMW",
                        "color": "red",
                        "parts": ["engine", "steering wheel"],
                    }
                ],
            )
        ],
        attributes=[
            Text(id="type", description="The make or brand of the car"),
            Text(id="color", description="The color of the car"),
            parts,
        ],
    )

    # To do nested objects you need to specify encoder_or_encoder_class="json"
    text = "The blue jeep has rear view mirror, roof, windshield"

    # Changed the encoder to json
    chain = create_extraction_chain(llm, cars_schema, encoder_or_encoder_class="json")
    output = chain.predict_and_parse(text=text)["data"]

    printOutput(output)

    prompt = chain.prompt.format_prompt(text=text).to_string()

    print(prompt)

    schema = Object(
        id="forecaster",
        description=(
            "User is controling an app that makes financial forecasts. "
            "They will give a command to update a forecast in the future"
        ),
        attributes=[
            Text(
                id="year",
                description="Year the user wants to update",
                examples=[("please increase 2014's customers by 15%", "2014")],
                many=True,
            ),
            Text(
                id="metric",
                description="The unit or metric a user would like to influence",
                examples=[("please increase 2014's customers by 15%", "customers")],
                many=True,
            ),
            Text(
                id="amount",
                description="The quantity of a forecast adjustment",
                examples=[("please increase 2014's customers by 15%", ".15")],
                many=True,
            ),
        ],
        many=False,
    )
    chain = create_extraction_chain(llm, schema, encoder_or_encoder_class="json")
    output = chain.predict_and_parse(text="please add 15 more units sold to 2023")[
        "data"
    ]

    printOutput(output)


def main():
    print("hello main...")
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=2000,
        model_kwargs={"frequency_penalty": 0, "presence_penalty": 0, "top_p": 1.0},
    )

    schema = Object(
        id="player",
        description=(
            "User is controlling a music player to select songs, pause or start them or play"
            " music by a particular artist."
        ),
        attributes=[
            Text(
                id="song",
                description="User wants to play this song",
                examples=[],
                many=True,
            ),
            Text(
                id="album",
                description="User wants to play this album",
                examples=[],
                many=True,
            ),
            Text(
                id="artist",
                description="Music by the given artist",
                examples=[("Songs by paul simon", "paul simon")],
                many=True,
            ),
            Text(
                id="action",
                description="Action to take one of: `play`, `stop`, `next`, `previous`.",
                examples=[
                    ("Please stop the music", "stop"),
                    ("play something", "play"),
                    ("play a song", "play"),
                    ("next song", "next"),
                ],
            ),
        ],
        many=False,
    )

    chain = create_extraction_chain(llm, schema, encoder_or_encoder_class="json")
    response = chain.invoke("play songs by paul simon and led zeppelin and the doors")
    print(f"response is {response}")

    schema, validator = from_pydantic(MusicRequest)
    chain = create_extraction_chain(
        llm, schema, encoder_or_encoder_class="json", validator=validator
    )
    response = chain.invoke("stop the music now")
    print(f"response is {response}")


if __name__ == "__main__":
    load_dotenv()
    main()
    # another_sample()
