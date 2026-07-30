import streamlit as st
import pandas as pd
import json
import plotly.express as px
from transformers import pipeline

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Amazon Review Sentiment Pipeline",
    page_icon="📊",
    layout="wide"
)

# ============================================
# LOAD DATA (cached so it only loads once)
# ============================================
@st.cache_data
def load_data():
    reviews = pd.read_csv("data/reviews_for_powerbi.csv")
    reviews['review_date'] = pd.to_datetime(reviews['review_date'])
    reviews['review_year'] = reviews['review_date'].dt.year

    topics = pd.read_csv("data/topic_summary.csv")
    complaint_words = pd.read_csv("data/top_complaint_words.csv")
    complaint_words.columns = ['term', 'negative_freq', 'positive_freq', 'ratio']

    with open("data/topic_labels.json") as f:
        topic_labels = json.load(f)

    return reviews, topics, complaint_words, topic_labels

reviews, topics, complaint_words, topic_labels = load_data()

# ============================================
# LOAD MODEL (cached as a resource - loads once, stays warm)
# ============================================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🔮 Live Sentiment Predictor", "📈 Analytics Dashboard", "🔍 Explore the Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About this project**

An end-to-end pipeline analyzing 15,587 real Amazon Electronics reviews:
- SQL relational database design
- VADER vs. DistilBERT sentiment comparison
- LDA topic modeling
- Live sentiment prediction (this app!)

Built by **Aayurshi Gawande**
[GitHub](https://github.com/Aayurshi07)
""")

# ============================================
# PAGE 1: OVERVIEW
# ============================================
if page == "🏠 Overview":
    st.title("📊 Amazon Review Sentiment & Product Insights Pipeline")
    st.markdown("### An end-to-end data pipeline: SQL → NLP → Interactive Deployment")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews Analyzed", f"{len(reviews):,}")
    col2.metric("Unique Products", f"{reviews['parent_asin'].nunique():,}")
    pos_pct = (reviews['bert_label'] == 'POSITIVE').mean() * 100
    col3.metric("Positive Sentiment", f"{pos_pct:.1f}%")
    neg_pct = (reviews['bert_label'] == 'NEGATIVE').mean() * 100
    col4.metric("Negative Sentiment", f"{neg_pct:.1f}%")

    st.markdown("---")

    st.markdown("""
    ## Project Summary

    This project builds a complete pipeline for understanding customer sentiment from real Amazon product reviews:

    1. **Data Engineering** — Sourced 15,587 reviews + product metadata directly from the Amazon Reviews 2023 dataset,
       designed a normalized 3-table SQL schema (Products, Reviews, Customers), and wrote advanced analytical SQL queries
       (JOINs, GROUP BY, window functions, CASE WHEN logic).

    2. **Sentiment Analysis (Model Comparison)** — Compared a lexicon-based approach (VADER) against a transformer-based
       approach (DistilBERT). While VADER achieved higher raw accuracy (89.9%), DistilBERT achieved dramatically higher
       **recall on negative reviews (93% vs. 51%)** — critical for a business trying to actually catch dissatisfied customers.

    3. **Topic Modeling & Keyword Analysis** — Used LDA to discover 6 underlying complaint themes, comparative word-frequency
       analysis to find complaint-specific vocabulary (up to 9x more common in negative reviews), and network graph
       visualization of how complaint concepts relate to one another.

    4. **Live Deployment** — This interactive app, deployed publicly, lets you test the trained sentiment model on any
       text in real time, and explore the underlying dataset and findings interactively.

    👈 **Use the sidebar to explore the live predictor, dashboard, and dataset.**
    """)

# ============================================
# PAGE 2: LIVE SENTIMENT PREDICTOR
# ============================================
elif page == "🔮 Live Sentiment Predictor":
    st.title("🔮 Live Sentiment Predictor")
    st.markdown("""
    Type or paste any product review below, and the trained **DistilBERT** model
    (the same model used throughout this project's analysis) will classify its sentiment in real time.
    """)

    with st.spinner("Loading DistilBERT model... (first load only, ~10-20 seconds)"):
        sentiment_model = load_model()

    example_reviews = {
        "— Select an example —": "",
        "Positive example": "I absolutely love this product! It works perfectly and the sound quality is amazing. Best purchase I've made this year.",
        "Negative example": "This broke after just two weeks of use. Customer support was unhelpful and I had to pay for return shipping myself. Would not recommend.",
        "Mixed/tricky example": "The camera quality is disappointing and the battery life is much shorter than advertised, but the price was low so I suppose I got what I paid for. Overall glad I bought it."
    }

    choice = st.selectbox("Try an example, or write your own below:", list(example_reviews.keys()))
    default_text = example_reviews[choice]

    user_text = st.text_area(
        "Enter review text:",
        value=default_text,
        height=150,
        placeholder="Type a product review here..."
    )

    if st.button("Analyze Sentiment", type="primary"):
        if user_text.strip() == "":
            st.warning("Please enter some review text first.")
        else:
            with st.spinner("Analyzing..."):
                result = sentiment_model(user_text)[0]
                label = result['label']
                score = result['score']

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if label == "POSITIVE":
                    st.success(f"### 😊 {label}")
                else:
                    st.error(f"### 😞 {label}")
            with col2:
                st.metric("Confidence", f"{score:.1%}")

            st.progress(float(score))
            st.caption(
                "Note: This model reads up to 512 tokens (~350-400 words) of text. "
                "Very long reviews are truncated, which can occasionally affect accuracy on mixed-sentiment text "
                "where the conclusion appears late in the review — a limitation discussed in the project README."
            )

# ============================================
# PAGE 3: ANALYTICS DASHBOARD
# ============================================
elif page == "📈 Analytics Dashboard":
    st.title("📈 Analytics Dashboard")

    # --- Filters ---
    categories = ["All Categories"] + sorted(reviews['main_category'].dropna().unique().tolist())
    selected_cat = st.selectbox("Filter by Category:", categories)

    filtered = reviews if selected_cat == "All Categories" else reviews[reviews['main_category'] == selected_cat]

    col1, col2, col3 = st.columns(3)
    col1.metric("Reviews in view", f"{len(filtered):,}")
    col2.metric("Avg. Star Rating", f"{filtered['rating'].mean():.2f}")
    neg_pct_f = (filtered['bert_label'] == 'NEGATIVE').mean() * 100
    col3.metric("Negative Sentiment %", f"{neg_pct_f:.1f}%")

    st.markdown("---")

    # --- Sentiment trend over time ---
    st.subheader("Sentiment Trend Over Time")
    trend = filtered[filtered['review_year'] >= 2008].groupby('review_year').apply(
        lambda x: pd.Series({
            'Positive %': (x['bert_label'] == 'POSITIVE').mean() * 100,
            'Negative %': (x['bert_label'] == 'NEGATIVE').mean() * 100,
            'Review Count': len(x)
        })
    ).reset_index()

    fig_trend = px.line(
        trend, x='review_year', y=['Positive %', 'Negative %'],
        markers=True, title="Positive vs. Negative Sentiment by Year (2008+)",
        labels={'value': 'Percentage', 'review_year': 'Year', 'variable': 'Sentiment'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Negative Sentiment by Category")
        cat_neg = reviews.groupby('main_category').apply(
            lambda x: pd.Series({
                'Negative %': (x['bert_label'] == 'NEGATIVE').mean() * 100,
                'Review Count': len(x)
            })
        ).reset_index()
        cat_neg = cat_neg[cat_neg['Review Count'] >= 30].sort_values('Negative %', ascending=False).head(10)

        fig_cat = px.bar(
            cat_neg, x='Negative %', y='main_category', orientation='h',
            title="Top 10 Categories by Negative Sentiment %",
            labels={'main_category': ''}
        )
        fig_cat.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.subheader("Complaint Topic Distribution")
        fig_topics = px.pie(
            topics, names='Topic', values='Review_Count',
            title="Negative Reviews by Complaint Theme (LDA Topics)",
            hole=0.4
        )
        st.plotly_chart(fig_topics, use_container_width=True)

    st.subheader("Top Complaint-Specific Words")
    st.caption("Words that appear disproportionately more often in negative vs. positive reviews")
    fig_words = px.bar(
        complaint_words.head(15).sort_values('ratio'),
        x='ratio', y='term', orientation='h',
        title="Top 15 Complaint Words (by Negative:Positive Frequency Ratio)",
        labels={'ratio': 'Frequency Ratio (Negative vs. Positive)', 'term': ''}
    )
    st.plotly_chart(fig_words, use_container_width=True)

# ============================================
# PAGE 4: EXPLORE THE DATA
# ============================================
elif page == "🔍 Explore the Data":
    st.title("🔍 Explore the Dataset")
    st.markdown("Browse a sample of the underlying review data used in this project.")

    col1, col2 = st.columns(2)
    with col1:
        cat_filter = st.multiselect(
            "Filter by category:",
            sorted(reviews['main_category'].dropna().unique().tolist())
        )
    with col2:
        sentiment_filter = st.multiselect(
            "Filter by sentiment:",
            ['POSITIVE', 'NEGATIVE']
        )

    display_df = reviews.copy()
    if cat_filter:
        display_df = display_df[display_df['main_category'].isin(cat_filter)]
    if sentiment_filter:
        display_df = display_df[display_df['bert_label'].isin(sentiment_filter)]

    st.caption(f"Showing {len(display_df):,} of {len(reviews):,} reviews")
    st.dataframe(
        display_df[['review_date', 'main_category', 'store', 'rating', 'bert_label', 'bert_score']].head(500),
        use_container_width=True
    )
