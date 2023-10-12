import re
from content.chapter_3.one_hierarchical_list_generation import openai_result


def test_heading_extraction():
    heading_pattern = r"\* (.+)"
    headings = re.findall(heading_pattern, openai_result)
    expected_headings = ["Introduction", "Efficient Data Management", "Conclusion"]
    assert headings == expected_headings


def test_subheading_extraction():
    subheading_pattern = r"\s+[a-z]\. (.+)"
    subheadings = re.findall(subheading_pattern, openai_result)
    expected_subheadings = [
        "Explanation of data engineering",
        "Importance of data engineering in today’s data-driven world",
        "Definition of data management",
        "How data engineering helps in efficient data management.",
        "Importance of Data Engineering in the modern business world",
        "Future of Data Engineering and its impact on the data ecosystem",
    ]
    assert subheadings == expected_subheadings
