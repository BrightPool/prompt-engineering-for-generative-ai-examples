import re


def test_heading_extraction():
    openai_result = """
    * Introduction
        a. Explanation of data engineering
        b. Importance of data engineering in today’s data-driven world
    * Efficient Data Management
        a. Definition of data management
        b. How data engineering helps in efficient data management.
    * Conclusion
        a. Importance of Data Engineering in the modern business world
        b. Future of Data Engineering and its impact on the data ecosystem
    """
    heading_pattern = r"\* (.+)"
    headings = re.findall(heading_pattern, openai_result)
    expected_headings = ["Introduction", "Efficient Data Management", "Conclusion"]
    assert headings == expected_headings


def test_subheading_extraction():
    openai_result = """
    * Introduction
        a. Explanation of data engineering
        b. Importance of data engineering in today’s data-driven world
    * Efficient Data Management
        a. Definition of data management
        b. How data engineering helps in efficient data management.
    * Conclusion
        a. Importance of Data Engineering in the modern business world
        b. Future of Data Engineering and its impact on the data ecosystem
    """
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
