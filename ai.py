import os
import re
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
    s = re.sub(r"[\d\u00BC-\u00BE\u2150-\u215E/.]+", " ", s)  # remove fractions/numbers
    s = re.sub(r"[^a-z\s]", " ", s)
    units = {"cup","cups","tablespoon","tablespoons","tbsp","teaspoon","teaspoons","tsp","grams","gram","g","kg","ounce","ounces","oz","ml","l","pinch","clove","cloves","slice","slices","package","can","cans","bunch"}
    stopwords = {"of","and","or","to","the","a","an","for","with","in","on","into","at","by","as","into","from"}
    parts = [p.strip() for p in re.split(r"[,\-\\/]", s) if p.strip()]
    for p in parts:
        words = [w for w in p.split() if w and w not in units and w not in stopwords]
        if not words:
            continue
        return " ".join(words)
    return s.strip()


def _extract_ingredients_from_text(text: str) -> list:
    if not text or not text.strip():
        return []
    t = text.replace('\r', '')
    # look for an 'ingredients' section
    m = re.search(r"ingredients\b([\s\S]{0,4000})", t, re.IGNORECASE)
    candidates = []
    if m:
        frag = m.group(1)
        # stop at common next-section headers
        stop = re.search(r"\n\s*(instructions|directions|method|preparation|steps)\b", frag, re.IGNORECASE)
        if stop:
            frag = frag[:stop.start()]
        lines = [ln.strip() for ln in frag.splitlines() if ln.strip()]
        candidates.extend(lines)
    # fallback: find lines that look like ingredient lines
    if not candidates:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        for ln in lines:
            if re.search(r"\b(cup|tbsp|tsp|teaspoon|tablespoon|g\b|gram|oz|ml|clove|pinch)\b", ln, re.IGNORECASE) or re.match(r"^[\-\*\d]", ln):
                candidates.append(ln)
            if len(candidates) > 60:
                break
    ingredients = set()
    for ln in candidates:
        # split by commas and connectors
        parts = re.split(r"[,;]\s*", ln)
        for p in parts:
            token = _normalize_ingredient_token(p)
            if token:
                ingredients.add(token)
    # final cleanup: remove very short tokens
    cleaned = [ing for ing in ingredients if len(ing) > 1]
    return cleaned


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
    """Find dishes from the uploaded recipe book and compute missing ingredients.

    Steps:
    1. Query the index for candidate dishes (grounded in the cookbook).
    2. For each dish, retrieve the recipe text from the index.
    3. Extract ingredient names heuristically from the recipe text.
    4. Compute missing items and match percentage against user's ingredients.
    """
    if query_engine is None:
        raise RuntimeError("No recipe book loaded. Upload a PDF first.")
    if not ingredients.strip():
        raise ValueError("Ingredients cannot be empty.")

    # 1) get candidate dish names (up to 8)
    prompt_list = f"""
Using ONLY the provided recipe book, list up to 8 dish NAMES (no numbers, no extra text) that can be made primarily with these ingredients: {ingredients}
Respond with one dish name per line. If none, output NO_DISHES_FOUND.
"""
    response = query_engine.query(prompt_list)
    raw = str(response).strip()
    if not raw or raw.upper().startswith("NO_DISHES_FOUND"):
        return []

    names = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # accept numbered or plain lines
        m = re.match(r"^\d+[\.)]?\s*(.+)$", line)
        if m:
            names.append(m.group(1).strip())
        else:
            names.append(line)
        if len(names) >= 8:
            break

    user_set = _normalize_user_ingredients(ingredients)

    results = []
    for name in names:
        # 2) retrieve recipe text for this dish
        try:
            recipe_text = get_recipe(name)
        except Exception:
            recipe_text = ''
        # detect missing recipe
        if not recipe_text or re.search(r"no recipe for|no recipe available|not found in the provided context", recipe_text, re.IGNORECASE):
            results.append({"name": name, "match": 0, "missing": []})
            continue
        # 3) extract ingredients
        recipe_ings = _extract_ingredients_from_text(recipe_text)
        recipe_ings_set = set(recipe_ings)
        # 4) compute missing
        if not recipe_ings_set:
            match_pct = 0
            missing = []
        else:
            # compare normalized tokens with some fuzzy matching
            have = set()
            for ui in user_set:
                # exact or partial match
                for ri in recipe_ings_set:
                    if ui in ri or ri in ui or ri.split()[-1] == ui.split()[-1]:
                        have.add(ri)
            common = have
            missing = sorted(list(recipe_ings_set - common))
            match_pct = round(100 * (len(common) / len(recipe_ings_set)))
        results.append({"name": name, "match": match_pct, "missing": missing})

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
"""
    response = query_engine.query(prompt)
    return str(response).strip()