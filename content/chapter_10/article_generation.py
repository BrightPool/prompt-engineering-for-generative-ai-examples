from langchain.chains import LLMChain
from typing import List, Dict, Any
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage


class OnlyStoreAIMemory(ConversationSummaryBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_ai_message(output_str)


class ContentGenerator:
    def __init__(
        self,
        topic: str,
        outline: Any,
        questions_and_answers: dict,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
    ):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.topic = topic
        self.outline = outline
        self.questions_and_answers = questions_and_answers

        prompt = f"""
        Act as a content SEO writer.
        You are currently writing a blog post on topic: {self.topic}.
        This is the outline of the blog post: {self.outline.json()}. You will be responsible for writing the blog post sections.
        ---
        Use your previous AI messages to avoid repeating yourself as you continually re-write the blog post sections.
        """
        chat = ChatOpenAI(model="gpt-3.5-turbo-16k")
        memory = OnlyStoreAIMemory(
            llm=chat,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=1200,
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        self.blog_post_chain = LLMChain(
            llm=chat, prompt=chat_prompt, memory=memory, output_key="blog_post"
        )
        self.chroma_db = None

    def split_and_vectorize_documents(self, text_documents):
        chunked_docs = self.text_splitter.split_documents(text_documents)
        self.chroma_db = Chroma.from_documents(chunked_docs, embedding=self.embeddings)  # type: ignore
        return self.chroma_db

    def generate_blog_post(self) -> List[str]:
        blog_post = []
        print("Generating the blog post...\n---")
        for subheading in self.outline.sub_headings:
            k = 5  # Initialize k
            while k >= 0:
                try:
                    relevant_documents = self.chroma_db.as_retriever().invoke(  # type: ignore
                        subheading.title, k=k
                    )
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    k -= 1
                if k < 0:
                    print(
                        "All attempts to fetch relevant documents have failed. Using an empty string for relevant_documents."
                    )
                    relevant_documents = ""
            section_prompt = f"""
            You are currently writing the section: {subheading.title}
            ---
            Here are the relevant documents for this section: {relevant_documents}.
            If the relevant documents are not useful, you can ignore them.
            You must never copy the relevant documents as this is plagiarism.
            --- 
            Here are the relevant insights that we gathered from our interview questions and answers: {self.questions_and_answers}. 
            You must include these insights where possible as they are important and will help our content rank better.

            ---
            You must follow the following principles:
            - You must write the section: {subheading.title}
            - Render the output in .md format
            - Include relevant formats such as bullet points, numbered lists, etc.
            ---
            Section text: 
            """
            result = self.blog_post_chain.predict(human_input=section_prompt)
            blog_post.append(result)

        print("Finished generating the blog post!\n---")
        return blog_post
