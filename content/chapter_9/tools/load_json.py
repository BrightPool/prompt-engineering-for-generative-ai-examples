from langchain.chat_models import ChatOpenAI
from typing import Optional, Type, Dict, Any
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import json
import os


class LoadJsonFileArgsSchema(BaseModel):
    """Input for loading a JSON file"""

    file_name: str = Field(..., description="The name of the JSON file to be loaded.")


class LoadJsonFileOutput(BaseModel):
    """Output for loading a JSON file"""

    status: str = Field(
        None, description="The status of the JSON file loading operation."
    )
    content: Dict[Any, Any] = Field(
        None, description="The content of the loaded JSON file."
    )


class LoadJsonFile(BaseTool):
    """Tool that loads a JSON file."""

    name: str = "load_json_file"
    args_schema: Type[BaseModel] = LoadJsonFileArgsSchema
    description: str = "Load a dictionary from a JSON file."
    return_direct: bool = False

    def _run(
        self,
        file_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs,
    ) -> LoadJsonFileOutput:
        try:
            with open(file_name, "r") as f:
                content = json.load(f)

            return LoadJsonFileOutput(status="success", content=content)

        except FileNotFoundError:
            return LoadJsonFileOutput(status=f"failed: File not found", content={})

        except json.JSONDecodeError:
            return LoadJsonFileOutput(status=f"failed: Invalid JSON format", content={})

        except Exception as e:
            return LoadJsonFileOutput(status=f"failed: {str(e)}", content={})

    async def arun(
        self,
        source_path: str,
        destination_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("arun not implemented")
