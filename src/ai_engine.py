"""GenAI Business Analyst: question -> Pandas evidence -> Gemini explanation (rule-based fallback if no key)."""
import os
import pandas as pd


def evidence(q, s, o, pay):
    q = q.lower()
    sel = s.groupby("seller_id").agg(revenue=("price", "sum"), items=("order_item_id", "count"),
                                     rating=("review_score", "mean"), late_rate=("late", "mean")).reset_index()
    if "seller" in q and any(w in q for w in ["rating", "low", "poor"]):
        t = sel[(sel["items"] >= 10) & (sel.revenue >= sel.revenue.quantile(.75))].sort_values("rating").head(10)
        return "Sellers in the top 25% by revenue, sorted by lowest average rating", t
    if "seller" in q:
        return "Top sellers by revenue", sel.sort_values("revenue", ascending=False).head(10)
    if "month" in q or "trend" in q:
        return "Sales by month", s.groupby("month").price.sum().reset_index().sort_values("price", ascending=False).head(12)
    if "categor" in q:
        return "Sales by category", s.groupby("category").agg(sales=("price", "sum"), items=("order_item_id", "count"),
                                                              rating=("review_score", "mean")).reset_index().sort_values("sales", ascending=False).head(10)
    if "product" in q:
        p = s.groupby("product_id").agg(sales=("price", "sum"), items=("order_item_id", "count"), rating=("review_score", "mean")).reset_index()
        if any(w in q for w in ["rating", "low", "poor"]):
            return "High-sales products (top 25%) with lowest ratings", p[(p["items"] >= 5) & (p.sales >= p.sales.quantile(.75))].sort_values("rating").head(10)
        return "Top products by sales", p.sort_values("sales", ascending=False).head(10)
    if "customer" in q:
        return "Top customers by number of orders", o.groupby("customer_unique_id").agg(orders=("order_id", "nunique")).reset_index().sort_values("orders", ascending=False).head(10)
    if "payment" in q or "installment" in q:
        return "Payment methods", pay.groupby("payment_type").agg(transactions=("order_id", "count"), value=("payment_value", "sum")).reset_index().sort_values("transactions", ascending=False)
    if any(w in q for w in ["late", "delay", "deliver"]) and any(w in q for w in ["review", "rating", "score", "associated"]):
        t = o.dropna(subset=["late", "review_score"]).groupby("late").review_score.agg(["mean", "count"]).reset_index()
        t["late"] = t.late.map({0.0: "on time", 1.0: "late"})
        return "Average review score: late vs on-time deliveries", t
    if any(w in q for w in ["state", "late", "deliver"]):
        return "Delivery performance by customer state", o.groupby("customer_state").agg(orders=("order_id", "count"), late_rate=("late", "mean"),
                                                                                         avg_days=("delivery_days", "mean")).reset_index().sort_values("late_rate", ascending=False).head(10)
    if any(w in q for w in ["review", "star", "rating"]):
        return "Review score distribution (%)", (o.review_score.round().value_counts(normalize=True).mul(100).round(1)
                                                 .rename("percent").rename_axis("score").reset_index().sort_values("score"))
    return "Headline KPIs", pd.DataFrame({"orders": [o.order_id.nunique()], "sales": [s.price.sum()],
                                          "avg_rating": [o.review_score.mean()], "late_rate": [o.late.mean()]})


def explain(q, title, ev, api_key=None):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if key:
        try:
            from google import genai
            prompt = (f"You are a business analyst for Olist, a Brazilian e-commerce marketplace. Manager question: {q}\n"
                      f"Actual data ({title}):\n{ev.round(2).to_csv(index=False)}\n"
                      "Answer ONLY from this data. Give: 1) a 2-sentence explanation, 2) three concise management recommendations. "
                      "Note that the dataset has no profit or cost data.")
            r = genai.Client(api_key=key).models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return r.text, "Gemini"
        except Exception as e:
            return f"Gemini call failed ({e}). Showing rule-based summary.\n\n" + _rule(title, ev), "rule-based"
    return _rule(title, ev), "rule-based (add a Gemini key for full AI analysis)"


def _rule(title, ev):
    top = ev.iloc[0].to_dict()
    row = ", ".join(f"{k} = {round(v, 2) if isinstance(v, float) else v}" for k, v in top.items())
    return (f"**{title}.** The leading row is: {row}. Compare the top rows against the overall average "
            "and prioritise follow-up on the outliers.")
