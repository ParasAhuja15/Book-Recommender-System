# Book-Recommender-System
Book Recommender System using LLMs
# Semantic Book Recommender and Analytics

A complete, end-to-end mini-project that turns a raw books metadata dump into a polished, AI-powered semantic recommender with explainable emotion filters and a lightweight demo UI. Built for clarity, reproducibility, and recruiter-friendly review.

- Data: 7k+ books with titles, authors, categories, descriptions, ratings, etc.
- Modeling: text classification for Fiction/Nonfiction, sentence-level emotion analysis, vector search for semantic retrieval.
- App: Gradio dashboard with category and emotion controls, returning book covers and concise captions.
- Stack: Python, Pandas, Transformers, LangChain, Chroma, OpenAI embeddings, Gradio, KaggleHub.

***

## Highlights

- Clean, documented notebooks that show the full journey: exploration → feature engineering → classification → emotion modeling → vector search.
- Sentence-level emotion scoring that captures peaks across a description, enabling “Happy/Sad/Surprising/Angry/Suspenseful” filtered recs.
- Production-lean app structure: single-file Gradio app powered by a persisted vector DB; clear requirements and .env usage.
- Thoughtful UX details: humanized author formatting, safe fallbacks for missing covers, short teaser descriptions in results.

***

## Repository Structure

- data-exploration.ipynb — initial EDA and feature engineering on the 7k+ books dataset (age_of_book, missing_description, words_in_description, category rollups).
- text-classification.ipynb — supervised simplification of categories (Fiction/Nonfiction), dataset curation, quality checks, error analysis slices.
- vector-search.ipynb — builds semantic retrieval using LangChain + OpenAI embeddings + Chroma on tagged descriptions.
- sentiment-analysis.ipynb — sentence-level emotion inference using j-hartmann/emotion-english-distilroberta-base; aggregates max-per-emotion signals per book; exports augmented dataset.
- gradio-dashboard.py — production-lite app that:
  - Loads books_with_emotions.csv
  - Builds/loads Chroma DB from tagged_description
  - Retrieves semantically similar books to a free-text query
  - Optionally filters by simplified category and sorts by target emotion
  - Displays covers and concise captions in a gallery
- requirements.txt — pinned versions for full environment reproduction.
- cover-not-found.jpg — default fallback image for missing thumbnails.

Deliverables produced during the workflow (local artifacts):
- books_cleaned.csv
- books_with_categories.csv
- books_with_emotions.csv
- tagged_description.txt
- Chroma vector store folder (created at runtime)

***

## How It Works

1) Data preparation
- Import books.csv from Kaggle (7k+ items).
- Standardize fields (authors split/format, title+subtitle merge, description word counts).
- Create quality flags (missing_description).
- Compute simple features (age_of_book = current_year − published_year).

2) Category simplification
- Map hundreds of granular categories to a simple Fiction/Nonfiction target.
- Train/evaluate a text classification approach (title, description, categories) and write predicted simple_categories into the dataset.

3) Semantic retrieval
- Create tagged_description: prepend isbn13 to description to preserve ID through vector search.
- Split and index with LangChain + OpenAI Embeddings into a Chroma DB.
- Retrieve top-k nearest neighbors for any natural-language query.

4) Emotion modeling
- Run sentence-level emotion inference on each description with the distilroberta-based model.
- Aggregate per-book max scores for: anger, disgust, fear, joy, sadness, surprise, neutral.
- Store scores in books_with_emotions.csv for fast, deterministic sorting in the app.

5) App logic
- Accepts a user query, optional category, optional tone.
- Gets initial semantic candidates via vector search; narrows by category; sorts by chosen emotion score.
- Returns a clean gallery of large thumbnails with author-formatted captions and truncated descriptions.

***

## Quickstart

Prerequisites
- Python 3.11+ recommended
- An OpenAI API key (for embeddings). Save as OPENAI_API_KEY in a .env file at repo root.

Environment setup
- Create and activate a virtual environment
- pip install -r requirements.txt
- Add .env with:
  - OPENAI_API_KEY=your_key

Data and indices
- Run notebooks in this order (or use your own data pipeline):
  1) data-exploration.ipynb
  2) text-classification.ipynb
  3) vector-search.ipynb
  4) sentiment-analysis.ipynb
- Outputs expected by the app:
  - books_with_emotions.csv
  - tagged_description.txt
  - A Chroma DB folder (created automatically on first run)

Run the app
- python gradio-dashboard.py
- A local URL will open with:
  - Query box: “A story about forgiveness”
  - Category filter: “All” or simplified buckets (e.g., Fiction, Nonfiction, Children’s Fiction, etc.)
  - Tone filter: “All”, “Happy”, “Surprising”, “Angry”, “Suspenseful”, “Sad”

***

## Notable Implementation Details

- Robust ID handling: the isbn13 is injected into text chunks to reliably map vector hits back to rows.
- Humanized captions: joins author names cleanly, truncates description to a crisp teaser.
- Emotion-aware re-ranking: instead of mixing similarity and emotion in a single model, decouples retrieval (semantic) from ranking (emotion), keeping behavior transparent and tunable.
- Sensible fallbacks: missing thumbnails replaced with a neutral placeholder; missing docs don’t crash the app.

***

## Results and Examples

- Query: “forgiveness and redemption in a small town”
  - Returns literary fiction and memoirs thematically aligned; “Happy” emphasis surfaces higher-joy descriptions.
- Query: “fast-paced space opera with moral dilemmas”
  - Surfaces science-fiction titles with relevant themes; “Suspenseful” boosts items with high fear signal.
- Query: “intro ecology and sense of wonder”
  - Nonfiction and nature titles; “Happy” or “Surprising” highlights uplifting or curiosity-laden descriptions.

***

## Design Choices and Tradeoffs

- Emotion aggregation uses per-sentence maxima (peak emotion) to capture salient moments from blurbs; alternative pooling (mean/median) is easy to swap if you prefer smoother signals.
- OpenAI embeddings chosen for strong out-of-the-box retrieval quality; can be replaced with local models if needed.
- Gradio provides the fastest path to an interactive demo; for production, this can be moved to a FastAPI backend + minimal frontend.

***

## Extensibility

- Add new facets: page count, year range, minimum rating, or rating count thresholds.
- Multi-objective ranking: combine cosine similarity with a weighted emotion score.
- Explanations: display top emotional sentences that drove a recommendation.
- Diversification: apply MMR or category-aware sampling to avoid near-duplicate results.
- Cold start images: call external cover APIs when thumbnails are missing.

***

## Reproducibility and Performance

- All library versions are pinned.
- Notebooks include explicit processing steps and intermediate previews.
- Emotion inference across ~5k books runs in under an hour on Apple MPS; can be batched or cached for speed.

***

##

- Clear problem framing and an opinionated end-to-end solution.
- Careful data engineering and feature design.
- Practical ML: choosing simple, explainable steps that combine well.
- Product sense: small UX details that make recommendations feel polished and trustworthy.
- Clean code and documentation with obvious next steps for scaling.

***

## Setup FAQs

- I don’t have OpenAI access. Replace embeddings in vector-search.ipynb and gradio-dashboard.py with a local alternative (e.g., sentence-transformers/all-MiniLM-L6-v2) via LangChain’s community embeddings.
- Where do CSVs come from? The exploration and modeling notebooks produce the derived CSVs used by the app; paths are relative to repo root.
- I see broken covers. The app automatically falls back to cover-not-found.jpg; ensure that file is present at repo root.

***
