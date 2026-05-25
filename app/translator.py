import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

SYSTEM_PROMPT = """
You are an intelligent retail search translator.

Your task:
Convert the user's query into EXACTLY 3 concise product search intents.

Rules:
- Automatically infer category: Fashion, Electronics, or Food & Beverages
- Use product-focused language (not advice)
- Include brand/type/spec where appropriate
- Do NOT assume gender unless explicitly mentioned
- Do NOT over-generalize
- Do NOT explain anything
- Output ONLY valid JSON

JSON format:
{
  "search_terms": ["...", "...", "..."]
}
"""

def translate_query(user_query: str):
    prompt = SYSTEM_PROMPT + f'\nUser query: "{user_query}"'

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json"
        }
    )

    text = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)["search_terms"]
