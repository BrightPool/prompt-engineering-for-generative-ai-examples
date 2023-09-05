from typing import Optional, Type, Callable, List
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import json


class ArgsSchema(BaseModel):
    """Input for Args"""

    questions: List[str] = Field(None, description="The questions.")


def _print_func(text: str) -> None:
    print("\n")
    print(text)


class HumanInTheLoop(BaseTool):
    """A Tool that asks user for input."""

    name: str = "human_in_the_loop_answer_question"
    description: str = (
        "You can ask a human for guidance when you need to answer a question. "
    )
    prompt_func: Callable[[str], None] = Field(default_factory=lambda: _print_func)
    input_func: Callable = Field(default_factory=lambda: input)
    args_schema: Type[BaseModel] = ArgsSchema
    return_direct: bool = False

    def _run(
        self,
        questions: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Human input tool."""
        questions_and_answers = []
        for question in questions:
            self.prompt_func(question)
            answer = self.input_func()
            questions_and_answers.append({"question": question, "answer": answer})
        with open("questions_and_answers.json", "w") as f:
            json.dump(questions_and_answers, f)
        return "Your output has been saved to a file called questions_and_answers.json."
