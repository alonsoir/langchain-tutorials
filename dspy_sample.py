import dspy

llm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=openai_key)

dspy.settings.configure(lm=llm)

# Implementation of a lie detector without optimization

text = "Barack Obama was not President of the USA"

lie_detector = dspy.Predict("text -> veracity")

response = lie_detector(text=text)

print(response.veracity)

# Let's say you want to control the output so that it is always a Boolean (True or False)
# The previous naive implementation does not guarantee that
# One way to guarantee that could be to use a more precise signature

# Precise signature


class LieSignature(dspy.Signature):
    """Identify if a statement is True or False"""

    text = dspy.InputField()
    veracity = dspy.OutputField(desc="a boolean 1 or 0")


lie_detector = dspy.Predict(LieSignature)

response = lie_detector(text=text)

print(response.veracity)

# Generate synthetic data

from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=1, api_key=openai_key)


class Data(BaseModel):
    fact: str = Field(
        description="A general fact about life or a scientific fact or a historic fact"
    )
    answer: str = Field(description="The veracity of a fact is a boolean 1 or 0")


parser = JsonOutputParser(pydantic_object=Data)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": "Generate data"})

# Create a list of 10 fact-answer pairs

list_of_facts = [chain.invoke({"query": "Generate data"}) for i in range(10)]

few_shot_examples = [dspy.Example(fact) for fact in list_of_facts]

print(list_of_facts)

# Problem with the previous approach, not enough data variability

# Access the schema
data_schema = Data.schema()

# Access the properties within the schema to get to the descriptions
fact_description = data_schema["properties"]["fact"]["description"]
answer_description = data_schema["properties"]["answer"]["description"]

list_of_facts = []

for i in range(10):
    prompt = f"Generate data. Should be different than {list_of_facts}. Answers should be diverse and representative of {answer_description}"
    example = chain.invoke({"query": prompt})
    list_of_facts.append(example)

few_shot_examples = [dspy.Example(fact) for fact in list_of_facts]

print(list_of_facts)

# Synthetic Prompt Optimization
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match

text = "Barack Obama was not President of the USA"

# define the fact as input to the lie detector
trainset = [x.with_inputs("fact") for x in few_shot_examples]


# define the signature to be used in by the lie detector module
# for the evaluation, you need to define an answer field
class Veracity(dspy.Signature):
    "Evaluate the veracity of a statement"
    fact = dspy.InputField(desc="a statement")
    answer = dspy.OutputField(desc="an assessment of the veracity of the statement")


class lie_detector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lie_identification = dspy.ChainOfThought(Veracity)

    def forward(self, fact):
        return self.lie_identification(fact=fact)


teleprompter = BootstrapFewShot(metric=answer_exact_match)

compiled_lie_detector = teleprompter.compile(lie_detector(), trainset=trainset)

response = compiled_lie_detector(fact=text)

print(f"veracity {response.answer}")
