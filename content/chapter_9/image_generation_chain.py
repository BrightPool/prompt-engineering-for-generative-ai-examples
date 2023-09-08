import base64
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
import os
import requests
import uuid

engine_id = "stable-diffusion-xl-1024-v1-0"
api_host = os.getenv("API_HOST", "https://api.stability.ai")
api_key = os.getenv("STABILITY_API_KEY", "INSERT_YOUR_IMAGE_API_KEY_HERE")

if api_key == "INSERT_YOUR_IMAGE_API_KEY_HERE":
    raise Exception(
        "You need to insert your API key in the image_generation_chain.py file. "
        "You can get your API key from https://platform.openai.com/"
    )


def create_image(title) -> str:
    chat = ChatOpenAI()
    # 1. Generate the image prompt:
    image_prompt = chat(
        [
            SystemMessage(
                content=f"""Create an image prompt that will be used for MidJourney for {title}."""
            )
        ]
    ).content

    # 2. Generate the image:
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "text_prompts": [
                {
                    "text": f"an illustration of {image_prompt} in the style of corporate memphis, white background, professional, clean lines, warm pastel colors"
                }
            ],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    image_paths = []

    for i, image in enumerate(data["artifacts"]):
        filename = f"{uuid.uuid4().hex[:7]}.png"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(image["base64"]))
        image_paths.append(filename)
    return image_paths
