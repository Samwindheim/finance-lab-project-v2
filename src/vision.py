"""
This is a script to run vision model capabilities for data extraction from images.
"""
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import mimetypes
import config

# --- Constants ---
PROMPTS_DIR = config.PROMPTS_DIR
GEMINI_MODEL = config.GEMINI_MODEL
GEMINI_API_BASE_URL = config.GEMINI_API_BASE_URL
# -----------------

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_json_from_image(image_path: str, extraction_type: str):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=GEMINI_API_BASE_URL,
    )

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return None

    prompt_file_path = os.path.join(PROMPTS_DIR, f"{extraction_type}.txt")
    if not os.path.exists(prompt_file_path):
        print(f"Error: Prompt file not found for extraction type '{extraction_type}' at '{prompt_file_path}'")
        return None

    base64_image = encode_image(image_path)
    mime_type = mimetypes.guess_type(image_path)[0]

    with open(prompt_file_path, "r") as f:
        prompt_text = f.read()

    print("\nSending request to Gemini...")

    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Example usage:
    # This allows running the script directly for testing
    test_image_path = "output_images/QCORE_2025‑08‑08_Memorandum_page_13.png"
    json_data = get_json_from_image(test_image_path, "underwriters")
    if json_data:
        print("\n--- Gemini Response ---")
        print(json_data)
        print("---------------------\n")
