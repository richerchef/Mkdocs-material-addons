import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def label_clusters_interactively(df, text_col='clean', cluster_col='cluster', label_col='system', n_terms=5, n_examples=5):
    """
    Interactively label clusters in a DataFrame based on common words and examples.
    Works in a Jupyter Notebook.
    """
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[text_col])
    terms = np.array(vectorizer.get_feature_names_out())

    labels = {}

    for cluster_id in sorted(df[cluster_col].unique()):
        # Skip already-labeled clusters
        if cluster_id in labels:
            continue

        cluster_docs = X[df[cluster_col] == cluster_id]
        cluster_mean = np.asarray(cluster_docs.mean(axis=0)).ravel()
        top_indices = cluster_mean.argsort()[::-1][:n_terms]
        top_words = terms[top_indices]

        print(f"\n=== Cluster {cluster_id} ===")
        print(f"Top {n_terms} words: {', '.join(top_words)}\n")

        # Show example rows
        examples = df.loc[df[cluster_col] == cluster_id, text_col].head(n_examples).tolist()
        for i, ex in enumerate(examples, 1):
            print(f"{i}. {ex}")

        # Prompt user for label
        label = input("\nEnter system label (or leave blank to skip): ").strip()
        if label:
            labels[cluster_id] = label
        else:
            labels[cluster_id] = None

    # Apply labels to dataframe
    df[label_col] = df[cluster_col].map(labels)
    print("\n✅ Labeling complete.")
    return df, labels
