# Standard libraries
from pydantic.v1 import BaseModel, Field
from typing import List, Any

# Langchain libraries
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import RunnableParallel


class Question(BaseModel):
    """Single Output - A question with no answer"""

    question: str = Field(None, description="An interview question to ask.")
    answer: None = None


class InterviewQuestions(BaseModel):
    """Output for Interview questions"""

    questions: List[Question] = Field(
        ..., min_items=5, max_items=5, description="List of interview questions."
    )


class InterviewChain:
    def __init__(self, topic: str, document_summaries: Any):
        self.topic = topic
        self.llm = ChatOpenAI(temperature=0)
        self.document_summaries = document_summaries

    def __call__(self) -> Any:
        # Create an LLM:
        model = ChatOpenAI(temperature=0.6)

        # Set up a parser + inject instructions into the prompt template:
        parser: PydanticOutputParser = PydanticOutputParser(
            pydantic_object=InterviewQuestions
        )

        system_message = """You are a content SEO researcher. Previously you have summarized and extracted key points from SERP results. 
        The insights gained will be used to do content research and we will compare the key points, insights and summaries across multiple articles.
        You are now going to interview a content expert. You will ask them questions about the following topic: {topic}.

        You must follow the following rules:
        - Return a list of questions that you would ask a content expert about the topic.
        - You must ask at least and at most 5 questions.
        - You are looking for information gain and unique insights that are not already covered in the {document_summaries} information.
        - You must ask questions that are open-ended and not yes/no questions.
        {format_instructions}
        """

        system_prompt = SystemMessagePromptTemplate.from_template(system_message)
        human_prompt = HumanMessagePromptTemplate.from_template(
            """Give me the first 5 questions"""
        )

        # Create the prompt:
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        chain = (
            RunnableParallel(
                topic=lambda x: self.topic,
                document_summaries=lambda x: self.document_summaries,
                format_instructions=lambda x: parser.get_format_instructions(),
            )
            | prompt
            | model
        )

        # Run the chat:
        result = chain.invoke(
            {
                "topic": self.topic,
                "document_summaries": self.document_summaries,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # Parse the llm response::
        return parser.parse(result.content)
