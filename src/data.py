"""Load, clean and join the Olist data. Falls back to a synthetic sample if data/ has no CSVs."""
import os
import numpy as np
import pandas as pd

D = "data"
FILES = {"customers": "olist_customers_dataset.csv", "order_items": "olist_order_items_dataset.csv",
         "payments": "olist_order_payments_dataset.csv", "reviews": "olist_order_reviews_dataset.csv",
         "orders": "olist_orders_dataset.csv", "products": "olist_products_dataset.csv",
         "sellers": "olist_sellers_dataset.csv", "cat": "product_category_name_translation.csv"}
DATES = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
         "order_delivered_customer_date", "order_estimated_delivery_date"]


def sample():
    r = np.random.default_rng(7)
    n = 5000
    states = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "PE", "CE"]
    cats = ["health_beauty", "watches_gifts", "bed_bath_table", "sports_leisure", "furniture_decor",
            "computers_accessories", "toys", "telephony", "housewares", "auto"]
    cu = pd.DataFrame({"customer_id": [f"c{i}" for i in range(n)],
                       "customer_unique_id": [f"u{i}" for i in r.integers(0, int(n * .9), n)],
                       "customer_city": r.choice(["sao paulo", "rio de janeiro", "belo horizonte", "curitiba"], n),
                       "customer_state": r.choice(states, n, p=[.4, .13, .12, .07, .07, .06, .05, .04, .03, .03])})
    p = pd.Timestamp("2017-01-01") + pd.to_timedelta(r.integers(0, 600 * 86400, n), unit="s")
    est = p + pd.to_timedelta(r.integers(15, 35, n), unit="D")
    dl = p + pd.to_timedelta(r.gamma(4, 3.5, n) * 86400, unit="s")
    stt = r.choice(["delivered", "shipped", "canceled", "unavailable"], n, p=[.97, .01, .01, .01])
    o = pd.DataFrame({"order_id": [f"o{i}" for i in range(n)], "customer_id": cu.customer_id, "order_status": stt,
                      "order_purchase_timestamp": p, "order_approved_at": p + pd.Timedelta(hours=2),
                      "order_delivered_carrier_date": p + pd.Timedelta(days=2),
                      "order_delivered_customer_date": dl.where(stt == "delivered"), "order_estimated_delivery_date": est})
    se = pd.DataFrame({"seller_id": [f"s{i}" for i in range(60)], "seller_city": "sao paulo", "seller_state": r.choice(states, 60)})
    pr = pd.DataFrame({"product_id": [f"p{i}" for i in range(250)], "product_category_name": r.choice(cats, 250)})
    k = r.integers(1, 4, n)
    oi = pd.DataFrame({"order_id": np.repeat(o.order_id.values, k)})
    oi["order_item_id"] = oi.groupby("order_id").cumcount() + 1
    oi["product_id"] = r.choice(pr.product_id, len(oi))
    oi["seller_id"] = r.choice(se.seller_id, len(oi))
    oi["price"] = np.round(r.lognormal(4.3, .8, len(oi)), 2)
    oi["freight_value"] = np.round(r.uniform(8, 35, len(oi)), 2)
    late = ((dl > est) & (stt == "delivered")).to_numpy() if hasattr(dl > est, "to_numpy") else (dl > est) & (stt == "delivered")
    sc = np.clip(np.round(r.normal(4.3, .9, n) - late * 1.6), 1, 5).astype(int)
    rv = pd.DataFrame({"review_id": range(n), "order_id": o.order_id, "review_score": sc,
                       "review_creation_date": dl, "review_answer_timestamp": dl + pd.Timedelta(days=2)})
    pay = pd.DataFrame({"order_id": o.order_id, "payment_type": r.choice(["credit_card", "boleto", "voucher", "debit_card"], n, p=[.74, .19, .05, .02]),
                        "payment_installments": r.choice([1, 2, 3, 4, 6, 10], n, p=[.5, .12, .1, .1, .1, .08])})
    tot = oi.assign(v=oi.price + oi.freight_value).groupby("order_id").v.sum()
    pay["payment_value"] = pay.order_id.map(tot)
    ct = pd.DataFrame({"product_category_name": cats, "product_category_name_english": cats})
    return {"customers": cu, "order_items": oi, "payments": pay, "reviews": rv, "orders": o,
            "products": pr, "sellers": se, "cat": ct}


def load():
    real = os.path.exists(f"{D}/{FILES['orders']}")
    d = {k: pd.read_csv(f"{D}/{v}") for k, v in FILES.items()} if real else sample()
    o = d["orders"]
    for c in DATES:
        o[c] = pd.to_datetime(o[c], errors="coerce")
    o["delivery_days"] = (o.order_delivered_customer_date - o.order_purchase_timestamp).dt.total_seconds() / 86400
    o["delay_days"] = (o.order_delivered_customer_date - o.order_estimated_delivery_date).dt.total_seconds() / 86400
    o["late"] = (o.delay_days > 0).astype(float).where(o.order_delivered_customer_date.notna())
    o["month"] = o.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp()
    rev = d["reviews"].groupby("order_id").review_score.mean().rename("review_score")
    o = o.merge(d["customers"], on="customer_id", how="left").merge(rev, on="order_id", how="left")
    ocols = ["order_id", "order_status", "order_purchase_timestamp", "delivery_days", "late", "month",
             "customer_unique_id", "customer_city", "customer_state", "review_score"]
    s = (d["order_items"].merge(o[ocols], on="order_id", how="left")
         .merge(d["products"], on="product_id", how="left").merge(d["sellers"], on="seller_id", how="left")
         .merge(d["cat"], on="product_category_name", how="left"))
    s["category"] = s.product_category_name_english.fillna(s.product_category_name).fillna("unknown")
    return d, s, o, real
