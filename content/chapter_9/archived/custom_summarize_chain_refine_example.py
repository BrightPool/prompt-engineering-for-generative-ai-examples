import asyncio
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pydantic import BaseModel
from typing import Dict, List, Optional, Union


class DocumentSummary(BaseModel):
    concise_summary: str
    writing_style: str
    key_points: List[str]
    expert_opinions: Optional[List[str]] = None
    metadata: Dict[
        str, str
    ] = {}  # This comes natively from the LangChain document loader


async def create_summary_from_text(
    document: Document,
    parser: PydanticOutputParser,
    llm: ChatOpenAI,
    text_splitter: RecursiveCharacterTextSplitter,
) -> Union[DocumentSummary, None]:
    # Split the parent document into chunks:
    split_docs = text_splitter.split_documents([document])

    # If there are no documents, return None:
    if len(split_docs) == 0:
        return None

    # Run a refine summarization chain that extracts unique key points and opinions within an article:
    prompt_template = """Act as a content SEO researcher. You are interested in summarizing and extracting key points from the following text. 
    The insights gained will be used to do content research and we will compare the key points, insights and summaries across multiple articles.
    ---
    - You must analyze the text and extract the key points and opinions from the following text
    - You must extract the key points and opinions from the following text:
    {text}

    {format_instructions}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Refine template:
    refine_template = (
        "Your job is to produce a final summary.\n"
        "We have provided an existing summary, key points, and expert opinions up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing content (only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary.\n"
        "If the context isn't useful or does not provide additional key points or expert opinions, you must return the original summary."
        "{format_instructions}"
    )
    refine_prompt = PromptTemplate.from_template(refine_template)

    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )

    print("Summarizing the data!")
    summary_result = await chain._acall(
        inputs={
            "input_documents": split_docs,
            "format_instructions": parser.get_format_instructions(),
        },
    )

    document_summary = parser.parse(summary_result["output_text"])
    document_summary.metadata = document.metadata
    return document_summary


async def create_all_summaries(
    text_documents: List[Document],
    parser: PydanticOutputParser,
    llm: ChatOpenAI,
    text_splitter: RecursiveCharacterTextSplitter,
) -> List[DocumentSummary]:
    # Create an array of coroutines:
    tasks = [
        create_summary_from_text(document, parser, llm, text_splitter)
        for document in text_documents
    ]

    # Execute the tasks concurrently and gather all the results:
    results = await asyncio.gather(*tasks)

    # Filter out None values
    summaries = [summary for summary in results if summary is not None]

    if len(summaries) == 0:
        raise ValueError("No summaries were created!")

    return summaries
