from langchain.prompts import (
    PromptTemplate,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import uuid

temperature = 0.0
model = OpenAI(temperature=temperature, max_tokens=1000)  # type: ignore


class BusinessName(BaseModel):
    uuid: str = Field(default=uuid.uuid4())
    name: str = Field(description="The name of the business")


class BusinessNames(BaseModel):
    names: List[BusinessName] = Field(description="A list of business names")


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=BusinessNames)

base_prompt = """
Generate 3 business names for a new start up company in the {industry} industry.
---
You must follow the following principles:
{principles}
---
{format_instructions}
"""

prompt = PromptTemplate(
    template=base_prompt,
    input_variables=["industry", "principles"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

principles = """
- The name must be easy to remember.
- Use the {industry} industry and Company context to create an effective name.
- The name must be easy to pronounce.
- You must only return the name without any other text or characters.
- Avoid returning full stops, \n or any other characters.
- The maximum length of the name must be 10 characters.
"""

_input = prompt.format_prompt(industry="Data Science", principles=principles)
output = model(_input.to_string())
pydantic_schema = parser.parse(output)
print(pydantic_schema, "\n")
print(pydantic_schema.dict())


# Chat Model Output Parser:
chat = ChatOpenAI()  # type: ignore
template = """Generate 5 business names for a new start up company in the {industry} industry. 
You must follow the following principles: {principles}
{format_instructions}
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)


chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
chat_prompt.output_parser = PydanticOutputParser(pydantic_object=BusinessNames)

# Get a chat completion from the formatted messages
chat_response_output = chat(
    chat_prompt.format_prompt(
        principles=principles,
        industry="Data Science",
        format_instructions=parser.get_format_instructions(),
    ).to_messages()
)

pydantic_schema = parser.parse(chat_response_output.content)
print(pydantic_schema, "\n")
print(pydantic_schema.dict())
