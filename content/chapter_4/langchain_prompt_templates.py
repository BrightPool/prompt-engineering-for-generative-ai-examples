from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

template = """
You are a creative consultant brainstorming names for businesses.

You must follow the following principles:
{principles}

Please generate a numerical list of 5 - 7 catchy names for a start-up in the {industry} industry that deals with {context}?

Here is an example of the format:
1. Name1
2. Name2
3. Name3
4. Name4
5. Name5
"""

chat = ChatOpenAI()
system_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt])

formatted_prompt = chat_prompt.format(
    industry="medical",
    context="creating AI solutions by automatically summarizing patient records",
    principles="1. Each name should be short and easy to remember. 2. Each name should be easy to pronounce. 3. Each name should be unique and not already taken by another company.",
)

# Generate names
business_names = chat.invoke(formatted_prompt)
print(business_names.content)
