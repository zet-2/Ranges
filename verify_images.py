import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

def analyze_image(path, label):
    try:
        img = Image.open(path)
        prompt = f"Describe this poker seat image ({label}). specificially: 1. Stack Size number? 2. Bet amount number? 3. Are there cards? 4. What is the text name?"
        response = model.generate_content([prompt, img])
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
