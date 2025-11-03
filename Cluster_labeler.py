import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from difflib import get_close_matches
from IPython.display import display, Markdown

def interactive_system_labeller(df, text_col='title_clean', system_col='system', n_clusters=20, max_features=500):
    """
    Cluster unknown system entries and interactively assign labels with suggestions.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with text and system columns.
    text_col : str
        The text column to cluster on.
    system_col : str
        The column to populate with system names.
    n_clusters : int
        Number of clusters for KMeans.
    max_features : int
        Max features for TF-IDF.
    """

    # Filter only rows with missing system labels
    unknown_df = df[df[system_col].isna() | (df[system_col] == '')].copy()

    if unknown_df.empty:
        print("✅ No unknown system entries to label.")
        return df

    print(f"🧩 Clustering {len(unknown_df)} unknown entries...")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(unknown_df[text_col].fillna(""))

    # Dynamic cluster count heuristic
    n_clusters = min(n_clusters, max(2, len(unknown_df) // 5))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    unknown_df['cluster'] = kmeans.fit_predict(X)

    # Known systems for suggestions
    known_systems = sorted(df[system_col].dropna().unique().tolist())

    print(f"💡 Found {n_clusters} clusters to review.\n")

    for c in sorted(unknown_df['cluster'].unique()):
        cluster_data = unknown_df[unknown_df['cluster'] == c]
        texts = cluster_data[text_col].dropna().tolist()
        top_examples = "\n".join(f"- {t}" for t in texts[:10])  # first 10 examples

        # Find best match suggestion
        joined_text = " ".join(texts).lower()
        suggestion = None
        for s in known_systems:
            if s.lower() in joined_text:
                suggestion = s
                break
        if not suggestion:
            # fallback: fuzzy match based on most common words
            tokens = [t for t in joined_text.split() if len(t) > 2]
            guess = get_close_matches(" ".join(tokens[:5]), known_systems, n=1, cutoff=0.4)
            suggestion = guess[0] if guess else None

        display(Markdown(f"### 🔹 Cluster {c} ({len(cluster_data)} items)\n{text_col} examples:\n{top_examples}"))

        if suggestion:
            display(Markdown(f"**💭 Suggested system:** **{suggestion}**"))
        else:
            print("💭 No clear system suggestion found.")

        # User input
        user_label = input("Enter system label (Enter to accept suggestion, 's' to skip): ").strip()

        if user_label.lower() == 's':
            continue
        elif user_label == '' and suggestion:
            user_label = suggestion

        if user_label:
            mask = (df.index.isin(cluster_data.index))
            df.loc[mask, system_col] = user_label
            print(f"✅ Updated {mask.sum()} rows with system = '{user_label}'\n")

    print("🎯 Labeling complete.")
    return df
