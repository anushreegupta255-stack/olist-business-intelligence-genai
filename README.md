# Olist Business Intelligence + GenAI Dashboard
Interactive Streamlit dashboard (Pandas + Plotly) with a Gemini-powered GenAI Business Analyst, built on the Olist Brazilian e-commerce dataset.

## Business areas
Customer, Sales & Orders, Category, Product, Seller, Payment, Review, Delivery & Fulfilment, plus an Executive Dashboard and GenAI Analyst. Global filters: date range and customer state.

## Run
```
python -m venv .venv && .venv\Scripts\activate    # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# put the 9 Olist CSVs in data/ (Kaggle: "Brazilian E-Commerce Public Dataset by Olist")
streamlit run app.py
```
Without CSVs the app runs on a synthetic sample (a banner says so).

## GenAI setup
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your Google AI Studio key (or paste it in the sidebar). The key is git-ignored. Flow: question -> Pandas evidence -> Gemini explanation + recommendations. Without a key a rule-based summary is shown.

## Structure
`app.py` (UI/pages), `src/data.py` (load, clean, join), `src/ai_engine.py` (GenAI), `docs/` (ER, Use Case, DFD, FDD diagrams).

## Notes / limitations
Olist has no quantity, cost or profit fields, so "sales" = item price, and items sold = item rows. Late = delivered after estimated date.
