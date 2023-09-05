# Standard libraries
from typing import Any

# Langchain libraries
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
    AgentExecutor,
    OpenAIFunctionsAgent,
    OpenAIMultiFunctionsAgent,
)
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Custom tools
from tools.generate_interview_questions import GenerateInterviewQuestions
from tools.human_in_the_loop import HumanInTheLoop

# Constants
MEMORY_KEY = "chat_history"


class AgentSetup:
    def __init__(self, topic: str):
        self.topic = topic
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.tools = [GenerateInterviewQuestions(), HumanInTheLoop()]

        system_message_content = (
            f"You are a very powerful assistant and are responsible "
            f"for investigating the following topic: {self.topic}. "
            f"You are bad at extracting key points from articles and need help. "
            f"Also, you must let the human answer questions as this is their interview. "
            f"---"
        )

        system_message = SystemMessage(content=system_message_content)

        self.memory = ConversationBufferMemory(
            memory_key=MEMORY_KEY, return_messages=True
        )

        prompt = OpenAIMultiFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)],
        )

        self.agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True, memory=self.memory
        )

    def run(self, input_message: str) -> Any:
        return self.agent_executor.run(input_message)
