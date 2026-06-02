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


def find_dishes(ingredients: str) -> list[str]:
    if query_engine is None:
        raise RuntimeError("No recipe book loaded. Upload a PDF first.")
    if not ingredients.strip():
        raise ValueError("Ingredients cannot be empty.")

    prompt = f"""
The user has these ingredients: {ingredients}

From the recipe book, list up to 8 dishes that can be made using primarily these ingredients.
Respond ONLY with a numbered list of dish names, nothing else.
Example:
1. Dish Name
2. Dish Name
"""
    response = query_engine.query(prompt)
    raw = str(response).strip()

    dishes = []
    for line in raw.split("\n"):
        match = re.match(r"^\d+[\.\)]\s*(.+)", line.strip())
        if match:
            dishes.append(match.group(1).strip())

    return dishes


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