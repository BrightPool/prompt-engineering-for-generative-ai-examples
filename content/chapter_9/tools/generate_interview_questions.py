from langchain.chat_models import ChatOpenAI
from typing import Optional, Type, List
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate
from langchain.tools import BaseTool


class ArgsSchema(BaseModel):
    """Input for Interview questions"""

    document_summaries: str = Field(..., description="The document summaries.")
    topic: str = Field(..., description="The topic to ask questions about.")


class Question(BaseModel):
    """Single Output - A question with no answer"""

    question: str = Field(None, description="An interview question to ask.")
    answer: None = None


class InterviewQuestions(BaseModel):
    """Output for Interview questions"""

    questions: List[Question]


class GenerateInterviewQuestions(BaseTool):
    """Tool that generates interview questions."""

    name: str = "generate_interview_questions"
    args_schema: Type[BaseModel] = ArgsSchema
    description: str = "Generate interview questions for a topic."
    return_direct: bool = False

    def _run(
        self,
        document_summaries: str,
        topic: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        # Create an LLM:
        chat = ChatOpenAI(temperature=0.6)

        print(kwargs)

        # Set up a parser + inject instructions into the prompt template:
        parser = PydanticOutputParser(pydantic_object=InterviewQuestions)
        system_message = """You are a content SEO researcher. Previously you have summarized and extracted key points from SERP results. 
        The insights gained will be used to do content research and we will compare the key points, insights and summaries across multiple articles.
        You are now going to interview a content expert. You will ask them questions about the following topic: {topic}.

        You must follow the following rules:
        - Return a list of questions that you would ask a content expert about the topic.
        - You must ask at least 5 questions.
        - You are looking for information gain and unique insights that are not already covered in the {document_summaries} information.
        - You must ask questions that are open-ended and not yes/no questions.
        {format_instructions}
        """
        system_prompt = SystemMessagePromptTemplate.from_template(system_message)
        system_message = system_prompt.format(
            document_summaries=document_summaries,
            topic=topic,
            format_instructions=parser.get_format_instructions(),
        )
        result = chat([system_message])
        return result.content

    async def arun(
        self,
        source_path: str,
        destination_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("arun not implemented")
