import re

# openai_result = generate_article_outline(prompt)

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

# Regular expression patterns
heading_pattern = r"\* (.+)"
subheading_pattern = r"\s+[a-z]\. (.+)"

# Extract headings and subheadings
headings = re.findall(heading_pattern, openai_result)
subheadings = re.findall(subheading_pattern, openai_result)

# Print results
print("Headings:\n")
for heading in headings:
    print(f"* {heading}")

print("\nSubheadings:\n")
for subheading in subheadings:
    print(f"* {subheading}")
