import asyncio
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic.v1 import BaseModel
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from typing import Dict, List, Optional, Union
import time
from langchain_core.prompts import PromptTemplate


class DocumentSummary(BaseModel):
    concise_summary: str
    writing_style: str
    key_points: List[str]
    expert_opinions: Optional[List[str]] = None
    metadata: Optional[
        Dict[str, str]
    ] = {}  # This comes natively from the LangChain document loader


async def create_summary_from_text(
    document: Document,
    parser: PydanticOutputParser,
    llm: ChatOpenAI,
    text_splitter: RecursiveCharacterTextSplitter,
) -> Union[DocumentSummary, None]:
    # Split the parent document into chunks:
    split_docs = text_splitter.split_documents([document])

    # If there are no documents after splitting, return None:
    if len(split_docs) == 0:
        return None

    # Get the first document, which will be the only document to be summarized:
    first_document = split_docs[0]

    # Run a refine summarization chain that extracts unique key points and opinions within an article:
    prompt_template = """Act as a content SEO researcher. You are interested in summarizing and extracting key points from the following text. 
    The insights gained will be used to do content research and we will compare the key points, insights and summaries across multiple articles.
    ---
    - You must analyze the text and extract the key points and opinions from the following text
    - You must extract the key points and opinions from the following text:
    {text}
    {format_instructions}
    """

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

    print("Summarizing the data!")

    # Start time:
    start_time = time.time()
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="text",
    )
    summary_result = await stuff_chain.ainvoke(
        {
            "input_documents": [first_document],
            "format_instructions": parser.get_format_instructions(),
        },
    )
    print(f"Time taken: {time.time() - start_time}")
    print("Finished summarizing the data!\n---" "")
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
