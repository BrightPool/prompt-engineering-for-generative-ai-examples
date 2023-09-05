import asyncio
from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
import os

# Custom imports:
from content_collection import collect_serp_data_and_extract_text_from_webpages
from custom_summarize_chain import create_all_summaries, DocumentSummary
from expert_interview_chain import InterviewChain
from article_outline_generation import BlogOutlineGenerator
from article_generation import ContentGenerator

os.environ["SERPAPI_API_KEY"] = ""


question_boxes = []
answer_boxes = []


def test():
    print("hello world")
    print(dir(question_boxes[0]))
    question_boxes[0].value = "Test"
    question_boxes[0].update(value="Test", visible=True)
    question_boxes[0].input.value = "Test"


def get_summary(topic):
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)

    try:
        result = new_loop.run_until_complete(async_get_summary(topic))
    finally:
        new_loop.close()

    return result


async def async_get_summary(topic):
    # Extract content from webpages into LangChain documents:
    text_documents = await collect_serp_data_and_extract_text_from_webpages(topic=topic)

    # Create summaries using LLM:
    llm = ChatOpenAI(temperature=0)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=400
    )
    parser = PydanticOutputParser(pydantic_object=DocumentSummary)
    # Create the summaries:
    print("Creating all of the summaries...\n---" "")
    summaries = await create_all_summaries(text_documents, parser, llm, text_splitter)

    # Create interview questions:
    print("Creating the interview questions...\n---" "")
    interview_chain = InterviewChain(topic=topic, document_summaries=summaries)
    interview_questions = interview_chain()

    # Update the gradio UI:
    for i, q in enumerate(interview_questions.questions):
        question_boxes[i].update(value=q.question, visible=True)
        answer_boxes[i].update(visible=True)

    return summaries, interview_questions


def generate_content(topic, summaries, text_documents):
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)

    try:
        result = new_loop.run_until_complete(
            async_generate_content(topic, summaries, text_documents)
        )
    finally:
        new_loop.close()

    return result


async def async_generate_content(
    topic,
    text_documents,
    summaries,
):
    # General Article Outline:
    blog_outline_generator = BlogOutlineGenerator(topic=topic)
    questions_and_answers = blog_outline_generator.questions_and_answers
    outline_result = blog_outline_generator.generate_outline(summaries)

    # Article Text Generation:
    content_gen = ContentGenerator(
        topic=topic, outline=outline_result, questions_and_answers=questions_and_answers
    )
    content_gen.split_and_vectorize_documents(text_documents)
    generated_text = content_gen.generate_blog_post()

    # Placeholder for image and prompt generation:
    generated_image = None
    generated_prompt = None

    return generated_text, generated_image, generated_prompt


with gr.Blocks() as demo:
    with gr.Row():
        topic = gr.Textbox(label="Topic", scale=85, value="Memetics")
        summarize_btn = gr.Button("Summarize and Generate Questions", scale=15)
        test_btn = gr.Button("Test", scale=15)

    with gr.Row():
        summaries = gr.Textbox(label="Summary", lines=10)
        interview_questions = gr.Textbox(label="Questions", lines=20)

    with gr.Row():
        summarize_btn.click(
            fn=get_summary,
            inputs=[topic],
            outputs=[summaries, interview_questions],
        )

    with gr.Row():
        test_btn.click(
            fn=test,
            inputs=[],
            outputs=[],
        )

    with gr.Column(scale=4):  # This column will take up 80% of the width
        question_boxes = []
        answer_boxes = []

        for i in range(5):
            with gr.Row():
                question = gr.Textbox(
                    label=f"Question {i + 1}", lines=1, visible=True, interactive=False
                )
                answer = gr.Textbox(label=f"Answer {i + 1}", lines=5, visible=True)
                question_boxes.append(question)
                answer_boxes.append(answer)

    # clear_btn = gr.Button("Clear", scale=15)
    # generate_btn = gr.Button("Generate", scale=30)

    # with gr.Row():
    #     with gr.Column():
    #         generated_content = gr.Textbox(label="Content", lines=50)
    #         generated_image = gr.Image(shape=(1200, 630))
    #         generated_prompt = gr.Textbox(label="Prompt")

    #     generate_btn.click(
    #         fn=generate_content,
    #         inputs=[topic, summaries, text_documents],  # Changed from 'summary_text'
    #         outputs=[generated_content, generated_image, generated_prompt],
    #     )

    #     # Reset the UI elements to default
    #     clear_btn.click(
    #         fn=lambda: ("", "", "", "", "", None, ""),  # Matching the output count
    #         inputs=[],
    #         outputs=[
    #             topic,
    #             summaries,
    #             text_documents,
    #             interview_questions,
    #             generated_content,
    #             generated_image,
    #             generated_prompt,
    #         ],
    #     )

demo.launch()
