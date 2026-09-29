import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data import load
from src import ai_engine

st.set_page_config("Olist Business Intelligence & GenAI", page_icon="🛒", layout="wide")
st.markdown("<style>[data-testid=stMetric]{background:#f4f6ff;border-radius:10px;padding:10px 14px;border:1px solid #c7d0ff}[data-testid=stMetric] *{color:#1e1b4b !important}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading Olist data...")
def get():
    return load()


d, S, O, real = get()
R = lambda x: f"R$ {x:,.0f}"
PAGES = ["Executive Dashboard", "Customer", "Sales & Orders", "Category", "Product", "Seller", "Payment", "Review", "Delivery & Fulfilment", "GenAI Business Analyst"]

with st.sidebar:
    st.title("🛒 Olist BI & GenAI")
    page = st.radio("Navigate", PAGES)
    st.divider()
    lo, hi = O.order_purchase_timestamp.min().date(), O.order_purchase_timestamp.max().date()
    dr = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    states = st.multiselect("Customer state", sorted(O.customer_state.dropna().unique()))
    if not real:
        st.warning("Showing SAMPLE data. Put the 9 Olist CSVs in /data for real results.")
if len(dr) < 2:
    st.stop()
a, b = pd.to_datetime(dr[0]), pd.to_datetime(dr[1]) + pd.Timedelta(days=1)


def flt(df):
    m = (df.order_purchase_timestamp >= a) & (df.order_purchase_timestamp < b)
    return df[m & df.customer_state.isin(states)] if states else df[m]


o, s = flt(O), flt(S)
pay = d["payments"][d["payments"].order_id.isin(o.order_id)]
dv = o[o.order_status == "delivered"]


def kpis(items):
    for c, (l, v) in zip(st.columns(len(items)), items):
        c.metric(l, v)


def show(fig):
    st.plotly_chart(fig, width="stretch")


def bar(df, x, y, title, h=False, color=None):
    fig = px.bar(df, x=y if h else x, y=x if h else y, orientation="h" if h else "v", title=title, color=color)
    if h:
        fig.update_yaxes(autorange="reversed")
    show(fig)


def insight(t):
    st.info("💡 **Business interpretation:** " + t)


sales, orders_n = s.price.sum(), o.order_id.nunique()
aov = (sales + s.freight_value.sum()) / max(orders_n, 1)
late = dv.late.mean() * 100
rating = o.review_score.mean()

if page == "Executive Dashboard":
    st.title("Olist Business Intelligence")
    st.caption("Explore. Analyse. Ask. Make better decisions.")
    kpis([("Customers", f"{o.customer_unique_id.nunique():,}"), ("Orders", f"{orders_n:,}"), ("Total Sales", R(sales)),
          ("Avg Order Value", f"R$ {aov:,.1f}"), ("Avg Rating", f"{rating:.2f} / 5"), ("Late Deliveries", f"{late:.1f}%")])
    m = s.groupby("month").agg(sales=("price", "sum"), orders=("order_id", "nunique")).reset_index()
    fig = go.Figure([go.Scatter(x=m.month, y=m.sales, name="Sales (R$)", fill="tozeroy"),
                     go.Scatter(x=m.month, y=m.orders, name="Orders", yaxis="y2")])
    fig.update_layout(title="Sales & Orders Trend", yaxis2=dict(overlaying="y", side="right"), legend_orientation="h")
    show(fig)
    c1, c2 = st.columns(2)
    with c1:
        bar(s.groupby("category").price.sum().nlargest(10).reset_index(), "category", "price", "Top 10 Categories by Sales (R$)", True)
    with c2:
        bar(s.groupby("seller_id").price.sum().nlargest(10).reset_index(), "seller_id", "price", "Top 10 Sellers by Sales (R$)", True)
    c3, c4 = st.columns(2)
    with c3:
        show(px.pie(o, names="order_status", hole=.5, title="Order Status Distribution"))
    with c4:
        g = go.Figure(go.Indicator(mode="gauge+number", value=100 - late, number={"suffix": "%"}, title={"text": "On-Time Delivery"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#16a34a"}}))
        show(g)
    insight("Use the sidebar filters to slice every page by date and state; ask the GenAI Analyst for deeper questions.")

elif page == "Customer":
    st.title("👥 Customer Intelligence")
    st.caption("Who are our most valuable customers, and how can we increase repeat purchases?")
    c = o.groupby("customer_unique_id").agg(orders=("order_id", "nunique"), first=("order_purchase_timestamp", "min"), last=("order_purchase_timestamp", "max")).reset_index()
    c = c.merge(s.groupby("customer_unique_id").price.sum().rename("revenue").reset_index(), on="customer_unique_id", how="left")
    rep = (c.orders > 1).mean() * 100
    kpis([("Unique Customers", f"{len(c):,}"), ("Repeat Rate", f"{rep:.1f}%"), ("Orders / Customer", f"{c.orders.mean():.2f}"), ("Revenue / Customer", f"R$ {c.revenue.mean():,.0f}")])
    c1, c2 = st.columns(2)
    with c1:
        bar(o.groupby("customer_state").customer_unique_id.nunique().reset_index(), "customer_state", "customer_unique_id", "Customers by State")
    with c2:
        nm = c.assign(m=c["first"].dt.to_period("M").dt.to_timestamp()).groupby("m").size().reset_index(name="new")
        show(px.line(nm, x="m", y="new", title="New Customers by Month"))
    c["recency_days"] = (o.order_purchase_timestamp.max() - c["last"]).dt.days
    show(px.histogram(c, x="recency_days", title="Customer Recency (days since last order)"))
    st.subheader("Top 15 customers by revenue")
    st.dataframe(c.nlargest(15, "revenue")[["customer_unique_id", "orders", "revenue"]], width="stretch")
    insight(f"Only {rep:.1f}% of customers buy again - retention campaigns for one-time buyers are the biggest growth lever.")

elif page == "Sales & Orders":
    st.title("📈 Sales & Order Intelligence")
    st.caption("Which products, categories and periods drive the highest sales?")
    kpis([("Orders", f"{orders_n:,}"), ("Product Sales", R(sales)), ("Freight", R(s.freight_value.sum())), ("AOV", f"R$ {aov:,.1f}"),
          ("Items Sold", f"{len(s):,}"), ("Items / Order", f"{len(s) / max(orders_n, 1):.2f}")])
    m = s.groupby("month").agg(sales=("price", "sum"), orders=("order_id", "nunique")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        show(px.line(m, x="month", y="sales", title="Revenue by Month"))
    with c2:
        show(px.line(m, x="month", y="orders", title="Orders by Month"))
    c3, c4 = st.columns(2)
    with c3:
        bar(s.groupby("customer_state").price.sum().reset_index().sort_values("price", ascending=False), "customer_state", "price", "Revenue by State")
    with c4:
        show(px.pie(o, names="order_status", hole=.5, title="Order Status"))
    insight("Compare peak months with category mix to plan inventory and promotions. Note: Olist has no profit data, so sales = product price.")

elif page == "Category":
    st.title("📦 Category Intelligence")
    cat = s.groupby("category").agg(sales=("price", "sum"), items=("order_item_id", "count"), avg_price=("price", "mean"),
                                    avg_freight=("freight_value", "mean"), rating=("review_score", "mean"),
                                    one_star=("review_score", lambda x: (x <= 1.5).mean() * 100)).reset_index()
    cat["share_%"] = cat.sales / cat.sales.sum() * 100
    kpis([("Categories", len(cat)), ("Top Category", cat.nlargest(1, "sales").category.iloc[0]), ("Top Share", f"{cat['share_%'].max():.1f}%")])
    show(px.treemap(cat, path=["category"], values="sales", color="rating", title="Category Sales (colour = avg rating)"))
    c1, c2 = st.columns(2)
    with c1:
        bar(cat.nlargest(10, "sales"), "category", "sales", "Top 10 by Revenue", True)
    with c2:
        bar(cat.nsmallest(10, "rating"), "category", "rating", "Lowest Rated Categories", True)
    st.dataframe(cat.sort_values("sales", ascending=False).round(2), width="stretch")
    insight("Grow high-share categories, and investigate categories with low ratings before scaling them.")

elif page == "Product":
    st.title("🏷️ Product Intelligence")
    p = s.groupby("product_id").agg(revenue=("price", "sum"), items=("order_item_id", "count"), rating=("review_score", "mean"), category=("category", "first")).reset_index()
    kpis([("Products Sold", f"{len(p):,}"), ("Avg Price", f"R$ {s.price.mean():,.1f}"), ("Avg Rating", f"{p.rating.mean():.2f}")])
    c1, c2 = st.columns(2)
    with c1:
        bar(p.nlargest(10, "revenue"), "product_id", "revenue", "Top 10 Products by Revenue", True)
    with c2:
        bar(p.nlargest(10, "items"), "product_id", "items", "Top 10 Products by Items Sold", True)
    show(px.scatter(p[p["items"] >= 3], x="revenue", y="rating", size="items", hover_name="product_id", color="category", title="Revenue vs Rating (bottom-right = high sales, poor rating)"))
    st.subheader("High-revenue, low-rating products")
    st.dataframe(p[(p.revenue >= p.revenue.quantile(.75)) & (p.rating < 3.5)].sort_values("rating"), width="stretch")
    insight("Products with strong sales but weak ratings deserve listing, quality or supplier review.")

elif page == "Seller":
    st.title("🏪 Seller Intelligence")
    st.caption("Which sellers drive revenue, and which need improvement?")
    sl = s.groupby("seller_id").agg(revenue=("price", "sum"), orders=("order_id", "nunique"), items=("order_item_id", "count"),
                                    rating=("review_score", "mean"), late_rate=("late", "mean"), state=("seller_state", "first")).reset_index()
    sl["late_rate"] *= 100
    kpis([("Active Sellers", f"{len(sl):,}"), ("Revenue / Seller", R(sl.revenue.mean())), ("Orders / Seller", f"{sl.orders.mean():.1f}"), ("Avg Seller Rating", f"{sl.rating.mean():.2f}")])
    c1, c2 = st.columns(2)
    with c1:
        bar(sl.nlargest(10, "revenue"), "seller_id", "revenue", "Top 10 Sellers by Revenue", True)
    with c2:
        bar(sl.groupby("state").size().reset_index(name="sellers"), "state", "sellers", "Sellers by State")
    show(px.scatter(sl, x="revenue", y="rating", size="orders", color="late_rate", hover_name="seller_id", color_continuous_scale="RdYlGn_r", title="Revenue vs Rating (colour = late %)"))
    st.subheader("Sellers needing attention: high revenue, low rating")
    st.dataframe(sl[(sl.revenue >= sl.revenue.quantile(.75)) & (sl["items"] >= 10)].nsmallest(10, "rating").round(2), width="stretch")
    insight("Prioritise support for high-revenue sellers with low ratings or high late rates - they carry the most customer-experience risk.")

elif page == "Payment":
    st.title("💳 Payment Intelligence")
    st.caption("How do customers pay, and how does it affect order value and instalments?")
    kpis([("Transactions", f"{len(pay):,}"), ("Total Payment Value", R(pay.payment_value.sum())), ("Avg Payment", f"R$ {pay.payment_value.mean():,.1f}"), ("Avg Instalments", f"{pay.payment_installments.mean():.1f}")])
    t = pay.groupby("payment_type").agg(transactions=("order_id", "count"), value=("payment_value", "sum"), avg=("payment_value", "mean")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        show(px.pie(t, names="payment_type", values="transactions", hole=.5, title="Payment Type Share"))
    with c2:
        bar(t, "payment_type", "avg", "Average Payment Value by Type")
    c3, c4 = st.columns(2)
    with c3:
        bar(pay.payment_installments.value_counts().sort_index().reset_index(), "payment_installments", "count", "Instalment Distribution")
    with c4:
        pm = pay.merge(o[["order_id", "month"]], on="order_id").groupby("month").payment_value.sum().reset_index()
        show(px.line(pm, x="month", y="payment_value", title="Payment Value by Month"))
    insight("Credit card dominates; higher instalment counts usually go with higher-value orders.")

elif page == "Review":
    st.title("⭐ Review Intelligence")
    st.caption("What drives customer satisfaction?")
    rs = o.review_score.dropna().round()
    kpis([("Avg Score", f"{rs.mean():.2f}"), ("5-Star %", f"{(rs == 5).mean() * 100:.1f}%"), ("1-Star %", f"{(rs == 1).mean() * 100:.1f}%"), ("Positive (4-5) %", f"{(rs >= 4).mean() * 100:.1f}%")])
    c1, c2 = st.columns(2)
    with c1:
        bar(rs.value_counts().sort_index().reset_index(), "review_score", "count", "Review Score Distribution")
    with c2:
        show(px.line(o.groupby("month").review_score.mean().reset_index(), x="month", y="review_score", title="Average Score by Month"))
    dd = dv.dropna(subset=["review_score", "delivery_days"]).copy()
    dd["bucket"] = pd.cut(dd.delivery_days, [0, 7, 14, 21, 30, 60, 500]).astype(str)
    bar(dd.groupby("bucket").review_score.mean().reset_index(), "bucket", "review_score", "Rating vs Delivery Time (days)")
    l = dd.groupby("late").review_score.mean()
    if len(l) == 2:
        st.metric("Avg score: on-time vs late", f"{l[0.0]:.2f}  vs  {l[1.0]:.2f}")
    insight("Reviews fall sharply as delivery time grows and when orders arrive late - fulfilment is the strongest satisfaction lever.")

elif page == "Delivery & Fulfilment":
    st.title("🚚 Delivery & Fulfilment")
    st.caption("Where are the major delays, and which sellers, categories and states are affected?")
    kpis([("Delivered", f"{len(dv):,}"), ("Avg Delivery Days", f"{dv.delivery_days.mean():.1f}"), ("Median Days", f"{dv.delivery_days.median():.1f}"),
          ("On-Time %", f"{100 - late:.1f}%"), ("Late %", f"{late:.1f}%"), ("Avg Delay (late orders)", f"{dv[dv.late == 1].delay_days.mean():.1f} d")])
    st_ = dv.groupby("customer_state").agg(late_rate=("late", "mean"), avg_days=("delivery_days", "mean"), orders=("order_id", "count")).reset_index()
    st_["late_rate"] *= 100
    c1, c2 = st.columns(2)
    with c1:
        bar(st_.sort_values("late_rate", ascending=False), "customer_state", "late_rate", "Late Delivery % by State")
    with c2:
        bar(st_.sort_values("avg_days", ascending=False), "customer_state", "avg_days", "Avg Delivery Days by State")
    show(px.histogram(dv, x="delivery_days", nbins=50, title="Delivery Time Distribution (days)"))
    c3, c4 = st.columns(2)
    sd = s[s.order_status == "delivered"]
    with c3:
        x = sd.groupby("seller_id").agg(late=("late", "mean"), n=("order_id", "nunique")).query("n>=10").nlargest(10, "late").reset_index()
        x["late"] *= 100
        bar(x, "seller_id", "late", "Sellers with Highest Late %", True)
    with c4:
        x = sd.groupby("category").late.mean().mul(100).nlargest(10).reset_index()
        bar(x, "category", "late", "Categories with Highest Late %", True)
    insight("Target the worst states and sellers first; a small drop in late % lifts ratings across the board.")

else:
    st.title("🤖 Olist GenAI Business Analyst")
    st.caption("Ask a business question. Pandas computes the evidence; Gemini explains it and recommends actions.")
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = None
    key = key or st.sidebar.text_input("Gemini API key (optional)", type="password")
    ex = ["Which sellers have high revenue but low ratings?", "Which month had the highest sales?", "Which category generated the highest sales?",
          "Which products have high sales but low ratings?", "Which payment method is most commonly used?", "What percentage of reviews are 1-star?",
          "Which states have the highest late delivery rate?", "Do late deliveries appear to be associated with lower review scores?",
          "Who are our top customers by number of orders?"]
    pick = st.selectbox("Example questions", ["(type my own)"] + ex)
    q = st.text_input("Ask your question", value="" if pick.startswith("(") else pick)
    if st.button("ANALYSE", type="primary") and q:
        title, ev = ai_engine.evidence(q, s, o, pay)
        st.subheader("📊 Data evidence")
        st.caption(title)
        st.dataframe(ev, width="stretch")
        with st.spinner("Analysing..."):
            txt, who = ai_engine.explain(q, title, ev, key)
        st.subheader("🤖 AI analysis & 💡 business insight")
        st.caption(f"Engine: {who}")
        st.markdown(txt)
