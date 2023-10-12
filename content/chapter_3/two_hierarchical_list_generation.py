import re

openai_result = """
* Introduction
  a. Explanation of data engineering
  b. Importance of data engineering in today’s data-driven world
* Efficient Data Management
    a. Definition of data management
    b. How data engineering helps in efficient data management.
    c. Why data engineering is important for data management
* Conclusion
    a. Importance of Data Engineering in the modern business world
    b. Future of Data Engineering and its impact on the data ecosystem
"""

section_regex = re.compile(r"\* (.+)")
subsection_regex = re.compile(r"\s*([a-z]\..+)")

result_dict = {}
current_section = None

for line in openai_result.split("\n"):
    section_match = section_regex.match(line)
    subsection_match = subsection_regex.match(line)

    if section_match:
        current_section = section_match.group(1)
        result_dict[current_section] = []
    elif subsection_match and current_section is not None:
        result_dict[current_section].append(subsection_match.group(1))

print(result_dict)
