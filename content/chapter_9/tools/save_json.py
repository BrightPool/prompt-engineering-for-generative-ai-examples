from typing import Optional, Type, Dict
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import json
import os


class SaveJsonFileArgsSchema(BaseModel):
    """Input for saving a JSON file"""

    content: Dict = Field(..., description="The dictionary to be saved as a JSON file.")
    file_name: Optional[str] = Field(
        None,
        description="The name of the JSON file where the dictionary should be saved.",
    )


class SaveJsonFileOutput(BaseModel):
    """Output for saving a JSON file"""

    status: str = Field(
        None, description="The status of the JSON file saving operation."
    )
    file_path: str = Field(None, description="The path where the JSON file is saved.")


class SaveJsonFile(BaseTool):
    """Tool that saves a JSON file."""

    name: str = "save_json_file"
    args_schema: Type[BaseModel] = SaveJsonFileArgsSchema
    description: str = "Save a dictionary to a JSON file."
    return_direct: bool = False

    def _run(
        self,
        content: Dict,
        file_name: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> SaveJsonFileOutput:
        # Generate default file name if none is provided
        if not file_name:
            file_name = "default_file.json"

        try:
            with open(file_name, "w") as f:
                json.dump(content, f)

            return SaveJsonFileOutput(
                status="success", file_path=os.path.abspath(file_name)
            )

        except Exception as e:
            return SaveJsonFileOutput(status=f"failed: {str(e)}", file_path="")

    async def arun(
        self,
        source_path: str,
        destination_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("arun not implemented")
