Here's a clean README for your project:

```markdown
# WhatsCooking AI 🍳

An AI-powered recipe assistant that lets you upload your own recipe book (PDF) and discover what you can cook based on ingredients you have on hand.

## Features

- 📄 Upload any recipe book as a PDF
- 🥦 Add ingredients from your pantry via a tag-based UI
- 🍽️ Get dish suggestions with ingredient match percentages
- 📋 View full recipes with SI unit conversions (°C, g, ml)
- 🎨 Animated glassmorphism UI

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI (Python) |
| AI / LLM | Groq (Llama 3.1 8B) via LlamaIndex |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| PDF Parsing | LlamaIndex `PDFReader` |
| Frontend | Vanilla HTML/CSS/JS |

## Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone https://github.com/your-username/whatscooking-ai.git
cd whatscooking-ai
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_key_here
```

### Run

```bash
uvicorn main:app --reload
```

The app opens in your browser automatically at `http://127.0.0.1:8000`.

## Usage

1. **Upload** your recipe book PDF
2. **Add ingredients** you have available
3. Click **Find Dishes** to get suggestions
4. Click any dish to expand its full recipe

## Project Structure

```
├── ai.py          # PDF loading, dish finding, recipe retrieval
├── main.py        # FastAPI routes and server setup
├── index.html     # Frontend UI
├── .env           # API keys (not committed)
└── requirements.txt
```

## Notes

- Only searches within your uploaded recipe book — no hallucinated recipes
- All measurements are auto-converted to SI units
- Requires an internet connection for the HuggingFace embedding model on first run
```
