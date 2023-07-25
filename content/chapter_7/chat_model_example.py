import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0)  # type: ignore
messages = [
    SystemMessage(content="Act as a senior software engineer at a startup company."),
    HumanMessage(
        content="Please can you provide a funny joke about software engineers?"
    ),
]
response = chat.predict_messages(messages=messages)
print(response.content)


text = "Translate this sentence from English to Spanish. I love software engineering."
response = chat.predict(text)
print(response)

# Using __call__:
response = chat(messages=messages)
print(response.content)


# Calling multiple sets of messages at once with .generate() and .agenerate():
synchronous_llm_result = chat.generate([messages, messages])
print(synchronous_llm_result)

asynchronous_llm_result = asyncio.run(chat.agenerate([messages, messages]))
print(asynchronous_llm_result)
