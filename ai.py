import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import PDFReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.program import LLMTextCompletionProgram

load_dotenv()

# Configure LLM globally with your current production key configuration
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("NEW_GROQ_API_KEY")
)

# Configure the embedding engine
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Clear global references for runtime state
query_engine = None
index_instance = None


# ─── Pydantic Schemas For Guaranteed API Structures ───
class DishAnalysis(BaseModel):
    dish: str = Field(description="The exact name of the dish as found within the recipe book text context.")
    match_percentage: int = Field(description="An integer from 0 to 100 calculating how well the user's available ingredients fulfill the requirements of this specific recipe.")
    missing: List[str] = Field(description="A clean list of necessary target ingredients for the recipe that were missing or not provided in the user's input list.")

class DishSuggestions(BaseModel):
    dishes: List[DishAnalysis] = Field(description="A list containing up to 8 matched dish options extracted from the recipe data.")


def load_pdf(file_path: str) -> str:
    """Parses incoming PDF text chunks and updates runtime query vector architectures."""
    global query_engine, index_instance
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target document not found at path: {file_path}")
        
    reader = PDFReader()
    documents = reader.load_data(file=file_path)
    
    # Generate vector tracking architectures
    index_instance = VectorStoreIndex.from_documents(documents)
    
    # Increase chunk lookup bounds (similarity_top_k=8) to ensure full recipes aren't truncated across pages
    query_engine = index_instance.as_query_engine(similarity_top_k=8)
    return "Recipe book loaded successfully."


def find_dishes(ingredients: str) -> list[dict]:
    """Uses semantic AI processing to extract matches, calculate missing assets, and format JSON."""
    if index_instance is None:
        raise RuntimeError("No recipe book loaded. Upload a PDF first.")
    if not ingredients.strip():
        raise ValueError("Ingredients cannot be empty.")

    # Drop complex regex string tokenizers. Let the LLM handle structural understanding.
    prompt_template = """
    You are an advanced culinary analysis system running on strict structural parameters.
    Review the provided recipe book context chunks to discover up to 8 dishes that can realistically be prepared using or adapting these available user pantry items: {user_ingredients}
    
    For each matched recipe found:
    1. Identify its full asset checklist from the book text.
    2. Assess the user's ingredients against that checklist.
    3. Generate a logical math match score percentage (0 to 100).
    4. Isolate individual ingredient names that are missing and list them cleanly.
    
    Context Source Text Material:
    {context_str}
    """
    
    # Initialize the structured conversion worker
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=DishSuggestions,
        prompt_template_str=prompt_template,
        llm=Settings.llm
    )
    
    # Manually extract relevant text segments matching user inventory variables
    retriever = index_instance.as_retriever(similarity_top_k=8)
    nodes = retriever.retrieve(ingredients)
    context_str = "\n\n".join([n.node.get_content() for n in nodes])
    
    try:
        # Run programmatic compilation forcing the structure
        structured_response: DishSuggestions = program(
            user_ingredients=ingredients,
            context_str=context_str
        )
        
        # Format the output into the standard dictionary list format expected by FastAPI
        return [item.model_dump() for item in structured_response.dishes]

    except Exception as e:
        print(f"[ERROR] Engine parsing block anomaly: {e}")
        raise ValueError("Failed to process target structural configurations safely from the model response.")


def get_recipe(dish_name: str) -> str:
    """Fetches the complete instruction manifest text block for a designated recipe name."""
    if query_engine is None:
        raise RuntimeError("No recipe book loaded. Upload a PDF first.")
    if not dish_name.strip():
        raise ValueError("Dish name cannot be empty.")

    prompt = f"""
    Retrieve the explicit cooking instructions and matching ingredients catalog for the dish: {dish_name}
    Include standard measurement metrics and structured breakdown steps as specified in the reference book text.
    Do not hallucinate external details. Return text based exclusively on factual context records.
    """
    response = query_engine.query(prompt)
    return str(response).strip()
