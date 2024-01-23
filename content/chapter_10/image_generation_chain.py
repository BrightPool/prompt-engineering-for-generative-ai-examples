import base64
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
import getpass
import os
import requests
import uuid

engine_id = "stable-diffusion-xl-1024-v1-0"
api_host = os.getenv("API_HOST", "https://api.stability.ai")
api_key = getpass.getpass("Enter your Stability API key: ")
os.environ["STABILITY_API_KEY"] = api_key


def create_image(title):
    # 1. Create the model:
    chat = ChatOpenAI()

    # 2. Generate the image prompt:
    image_prompt = chat.invoke(
        [
            SystemMessage(
                content=f"""Create an image prompt that will be used for MidJourney for {title}."""
            )
        ]
    ).content

    # 3. Generate the image:
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        timeout=60,
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
