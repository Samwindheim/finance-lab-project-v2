"""
Vision Model Interaction for Structured Data Extraction.

This module is responsible for interfacing with the Google Gemini vision model to extract structured data (JSON) from PDF page images. It takes one or more images, combines them with a textual prompt and page text, and sends them to the Gemini API. The primary function, get_json_from_image, orchestrates this process, handling API key management, request payload construction, and returning the raw JSON response from the model.
"""
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import mimetypes
import config
from typing import List

# --- Constants ---
PROMPTS_DIR = config.PROMPTS_DIR
GEMINI_MODEL = config.GEMINI_MODEL
GEMINI_API_BASE_URL = config.GEMINI_API_BASE_URL
# -----------------

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_json_from_image(image_paths: List[str], page_texts: str, extraction_type: str):
    load_dotenv()
    # (and text from the page)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=GEMINI_API_BASE_URL,
    )

    # --- Prepare content for multi-modal request ---
    content = []
    
    # Add the prompt text first
    prompt_file_path = os.path.join(PROMPTS_DIR, f"{extraction_type}.txt")
    if not os.path.exists(prompt_file_path):
        print(f"Error: Prompt file not found for extraction type '{extraction_type}' at '{prompt_file_path}'")
        return None
    with open(prompt_file_path, "r") as f:
        prompt_text = f.read()
    content.append({"type": "text", "text": prompt_text})

    # Add the extracted page text
    content.append({"type": "text", "text": "--- TEXT ---"})
    content.append({"type": "text", "text": page_texts})
    content.append({"type": "text", "text": "--- END TEXT ---"})

    # Add each image
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at '{image_path}'")
            return None
        
        base64_image = encode_image(image_path)
        mime_type = mimetypes.guess_type(image_path)[0]
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
        })

    print(f"\nSending request to Gemini with {len(image_paths)} image(s)...")
    # print page texts
    # print(f"\nPage texts: {page_texts}")

    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Example usage:
    # This allows running the script directly for testing
    test_image_path = "output_images/QCORE_2025‑08‑08_Memorandum_page_13.png"
    # A dummy text is added here for testing purposes
    dummy_text = "This is the extracted text from the page."
    json_data = get_json_from_image([test_image_path], dummy_text, "underwriters")
    if json_data:
        print("\n--- Gemini Response ---")
        print(json_data)
        print("---------------------\n")
