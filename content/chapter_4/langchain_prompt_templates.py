from langchain import PromptTemplate
from langchain.llms import OpenAI

template = """
You are a creative consultant brainstorming names for businesses.

You must follow the following principles:
{principles}

Please generate a numerical list of 5 catchy names for a start-up in the {industry} industry that deals with {context}?

Here is an example of the format:
1. Name1
2. Name2
3. Name3
4. Name4
5. Name5

"""

prompt = PromptTemplate.from_template(template)
formatted_prompt = prompt.format(
    industry="medical",
    context="creating AI solutions by automatically summarizing patient records",
    principles="1. Each name should be short and easy to remember. 2. Each name should be easy to pronounce. 3. Each name should be unique and not already taken by another company.",
)

llm = OpenAI(temperature=0.8)  # type: ignore
# Generate names
business_names = llm.predict(formatted_prompt)
print(business_names)
