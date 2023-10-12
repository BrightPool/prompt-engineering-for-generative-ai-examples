import json

# openai_json_result = generate_article_outline(prompt)

openai_json_result = """
{
    "Introduction": [
        "a. Overview of coding and programming languages",
        "b. Importance of coding in today's technology-driven world"],
    "Conclusion": [
        "a. Recap of the benefits of learning code",
        "b. The ongoing importance of coding skills in the modern world"]
}
"""

parsed_json_payload = json.loads(openai_json_result)
print(parsed_json_payload)
