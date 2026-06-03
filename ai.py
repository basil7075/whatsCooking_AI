import os
import re
import json
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import PDFReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

query_engine = None


def load_pdf(file_path: str) -> str:
    global query_engine
    reader = PDFReader()
    documents = reader.load_data(file=file_path)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=5)
    return "Recipe book loaded successfully."


def _normalize_ingredient_token(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[\d\u00BC-\u00BE\u2150-\u215E/.]+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    units = {"cup","cups","tablespoon","tablespoons","tbsp","teaspoon","teaspoons",
             "tsp","grams","gram","g","kg","ounce","ounces","oz","ml","l",
             "pinch","clove","cloves","slice","slices","package","can","cans","bunch"}
    stopwords = {"of","and","or","to","the","a","an","for","with","in","on","into","at","by","as","into","from"}
    words = [w for w in s.split() if w not in units and w not in stopwords]
    return " ".join(words).strip()


def _normalize_user_ingredients(s: str) -> set:
    parts = re.split(r"[,;]\s*", s.lower())
    normalized = set()
    for p in parts:
        p = re.sub(r"\(.*?\)", "", p)
        p = re.sub(r"[^a-z\s]", " ", p)
        p = p.strip()
        if not p:
            continue
        # take first 3 words
        words = [w for w in p.split() if w]
        if not words:
            continue
        normalized.add(" ".join(words[:3]))
    return normalized


def find_dishes(ingredients: str) -> list[dict]:
    if query_engine is None:
        raise RuntimeError("No recipe book loaded. Upload a PDF first.")
    if not ingredients.strip():
        raise ValueError("Ingredients cannot be empty.")

    prompt_list = f"""
Using ONLY the provided recipe book, list up to 5 dishes that can be made primarily with these ingredients: {ingredients}

You MUST output your response in strict JSON format. Do NOT include any reasoning, explanation, or chain-of-thought.
Return ONLY a JSON object with a single key "dishes", containing a list of objects with:
- "name": dish name (string)
- "ingredients": list of ingredient strings (just names, no quantities)

Example:
{{
  "dishes": [
    {{
      "name": "Tomato Pasta",
      "ingredients": ["tomato", "pasta", "garlic", "olive oil"]
    }}
  ]
}}
"""
    response = query_engine.query(prompt_list)
    raw = str(response).strip()

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        dishes_raw = parsed.get("dishes", [])
        seen = set()
        unique_dishes = []
        for d in dishes_raw:
            name = d.get("name", "").strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                unique_dishes.append(d)
        dishes_raw = unique_dishes
        print(f"[DEBUG] Parsed JSON response: {parsed}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        print(f"[ERROR] Raw response was: {raw}")
        raise ValueError("Failed to parse dish suggestions from AI. Please try again.")

    reasoning_phrases = ["okay, let's see", "the user wants me", "looking at page", "first, looking at", "wait, the context"]

    user_set = _normalize_user_ingredients(ingredients)
    results = []

    for dish in dishes_raw[:8]:
        name = dish.get("name", "")
        if not name or len(name) > 100:
            continue
        if any(phrase in name.lower() for phrase in reasoning_phrases):
            continue

        recipe_ings_set = set(_normalize_ingredient_token(i) for i in dish.get("ingredients", []) if i.strip())
        recipe_ings_set = {i for i in recipe_ings_set if len(i) > 1}

        if not recipe_ings_set:
            results.append({"dish": name, "match_percentage": 0, "missing": []})
            continue

        have = set()
        for ui in user_set:
            for ri in recipe_ings_set:
                if ui in ri or ri in ui or ri.split()[-1] == ui.split()[-1]:
                    have.add(ri)

        missing = sorted(list(recipe_ings_set - have))
        match_pct = round(100 * (len(have) / len(recipe_ings_set)))
        results.append({"dish": name, "match_percentage": match_pct, "missing": missing})

    return results

def get_recipe(dish_name: str) -> str:
    if query_engine is None:
        raise RuntimeError("No recipe book loaded. Upload a PDF first.")
    if not dish_name.strip():
        raise ValueError("Dish name cannot be empty.")

    prompt = f"""
From the recipe book, provide the complete recipe for: {dish_name}
Include ingredients with quantities and step-by-step cooking instructions.
Only use information from the recipe book.
Convert ALL measurements to SI units:
- Temperature in Celsius (°C)
- Weight in grams (g) or kilograms (kg)
- Volume in millilitres (ml) or litres (l)
- Length/size in centimetres (cm)
If the book uses Fahrenheit, cups, ounces, or inches — convert them. Show only SI values.
"""
    response = query_engine.query(prompt)
    return str(response).strip()