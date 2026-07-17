import os
import sys
import io
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=6500,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)
model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

def analyze_image(path, label):
    try:
        img = Image.open(path)
        prompt = f"Describe this poker seat image ({label}). specificially: 1. Stack Size number? 2. Bet amount number? 3. Are there cards? 4. What is the text name?"
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=85)
        response = client.models.generate_content(
            model=model,
            contents=[
                prompt,
                types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        print(f"--- {label} ({path}) ---")
        print(response.text.strip())
        print("-" * 30)
    except Exception as e:
        print(f"Error {label}: {e}")

def main():
    print("Verifying debug images content...")
    analyze_image("debug_images/hero.png", "Hero (Seat 5)")
    analyze_image("debug_images/seat4.png", "Seat 4")

if __name__ == "__main__":
    main()
