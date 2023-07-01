import asyncio
from langchain.llms import OpenAI
from typing import List

llm = OpenAI(temperature=0.9)  # type: ignore
response = llm.predict("Generate a name for a new start up company.")
print(response)


async def run_async_llm(llm: OpenAI, prompt: str) -> str:
    return await llm.apredict(prompt)


print(asyncio.run(run_async_llm(llm, "Generate a name for a new start up company.")))


# A different way to invoke the LLM class:
llm_response = llm(
    "Generate a name for a new start up company."
)  # llm.__call__("Generate a name for a new start up company.")
print("This LLM response is using the __call__ function", llm_response)


# Calling multiple prompts at once snychronously:
synchronous_llm_result = llm.generate(
    [
        "Generate a name for a new start up company in the medical profession",
        "Generate a name for a new start up company in  the financial sector",
    ]
)
print(synchronous_llm_result)
print(type(synchronous_llm_result))
print("------------------")
print(
    "These are the text generations:",
    [gen.text for sublist in synchronous_llm_result.generations for gen in sublist],
)
print("This is the llm output dictionary", synchronous_llm_result.llm_output)
print("A list of RunInfo objects", synchronous_llm_result.run)
if synchronous_llm_result.run:
    print("Run ids: ", [run.run_id for run in synchronous_llm_result.run])


# Calling multiple prompts at once asynchronously:
async def run_async_llms(llm: OpenAI, prompts: List[str]):
    return await llm.agenerate(prompts=prompts)


print(
    asyncio.run(
        run_async_llms(
            llm,
            [
                "Generate a name for a new start up company in the medical profession",
                "Generate a name for a new start up company in  the financial sector",
            ],
        )
    ),
)

# Improving the company name generator with a custom prompt:
industry = "medical"
custom_prompt = f"""
Generate a name for a new start up company in the {industry} profession.
Company context: The company will be creating AI solutions by automatically summarizing patient records.
---
You must follow the following principles:
- The name must be easy to remember.
- Use the {industry} industry and Company context to create an effective name.
- The name must be easy to pronounce.
- You must only return the name without any other text or characters.
- Avoid returning full stops, \n or any other characters.
- The maximum length of the name must be 10 characters.
---
Examples:
- Name: DigiMed
- Name: MedTech
- Name: MedX
---
Name: 
"""

llm = OpenAI(
    temperature=1.2,
    best_of=3,
)  # type: ignore
optimized_response = llm.predict(custom_prompt)
print(optimized_response)
