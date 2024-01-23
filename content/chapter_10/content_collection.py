from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
import os
import pandas as pd
from serpapi import GoogleSearch
from typing import List


class ChromiumLoader(AsyncChromiumLoader):
    async def load(self):
        raw_text = [await self.ascrape_playwright(url) for url in self.urls]
        # Return the raw documents:
        return [Document(page_content=text) for text in raw_text]


async def get_html_content_from_urls(
    df: pd.DataFrame, number_of_urls: int = 3, url_column: str = "link"
) -> List[Document]:
    # Get the HTML content of the first 3 URLs:
    urls = df[url_column].values[:number_of_urls].tolist()

    # If there is only one URL, convert it to a list:
    if isinstance(urls, str):
        urls = [urls]

    # Check for empty URLs:
    urls = [url for url in urls if url != ""]

    # Check for duplicate URLs:
    urls = list(set(urls))

    # Throw error if no URLs are found:
    if len(urls) == 0:
        raise ValueError("No URLs found!")
    # loader = AsyncHtmlLoader(urls) # Faster but might not always work.
    loader = ChromiumLoader(urls)
    docs = await loader.load()
    return docs


def extract_text_from_webpages(documents: List[Document]):
    html2text = Html2TextTransformer()
    return html2text.transform_documents(documents)


async def collect_serp_data_and_extract_text_from_webpages(
    topic: str,
) -> List[Document]:
    search = GoogleSearch(
        {
            "q": topic,
            "location": "Austin,Texas",
            "api_key": os.environ["SERPAPI_API_KEY"],
        }
    )
    # Get the results:
    result = search.get_dict()

    # Put the results in a Pandas DataFrame:
    serp_results = pd.DataFrame(result["organic_results"])

    # Extract the html content from the URLs:
    html_documents = await get_html_content_from_urls(serp_results)

    # Extract the text from the URLs:
    text_documents = extract_text_from_webpages(html_documents)

    return text_documents
