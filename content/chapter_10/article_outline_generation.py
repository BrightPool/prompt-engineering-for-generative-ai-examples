from typing import List, Any
from pydantic.v1 import BaseModel


class SubHeading(BaseModel):
    title: str


class BlogOutline(BaseModel):
    title: str
    sub_headings: List[SubHeading]


# Langchain libraries:
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai.chat_models import ChatOpenAI

# Custom types:
from custom_summarize_chain import DocumentSummary


class BlogOutlineGenerator:
    def __init__(self, topic: str, questions_and_answers: Any):
        self.topic = topic
        self.questions_and_answers = questions_and_answers

        # Create a prompt
        prompt_content = """
        Based on my answers and the summary, generate an outline for a blog article on {topic}.
        topic: {topic}
        document_summaries: {document_summaries}
        ---
        Here is the interview which I answered: {interview_questions_and_answers}
        ---
        Output format: {format_instructions}
        """

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            prompt_content
        )
        self.chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        # Create an output parser
        self.parser = PydanticOutputParser(pydantic_object=BlogOutline)

        # Set up the chain
        self.outline_chain = self.chat_prompt | ChatOpenAI() | self.parser

    def generate_outline(self, summaries: List[DocumentSummary]) -> Any:
        print("Generating the outline...\n---")
        result = self.outline_chain.invoke(
            {
                "topic": self.topic,
                "document_summaries": [s.dict() for s in summaries],
                "interview_questions_and_answers": self.questions_and_answers,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )
        print("Finished generating the outline!\n---")
        return result
