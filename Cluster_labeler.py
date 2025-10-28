import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

def label_clusters_interactively(
    df, 
    text_col='clean', 
    cluster_col='cluster', 
    label_col='system', 
    n_terms=5, 
    n_examples=5, 
    progress_file='cluster_labels_progress.csv'
):
    """
    Interactively label clusters in a DataFrame.
    - Saves progress to progress_file after each cluster
    - Can resume from existing progress_file
    """
    # Load existing progress if available
    if os.path.exists(progress_file):
        labels = pd.read_csv(progress_file, index_col=0)['label'].to_dict()
        print(f"Resuming from existing progress file: {progress_file}")
    else:
        labels = {}

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[text_col])
    terms = np.array(vectorizer.get_feature_names_out())

    for cluster_id in sorted(df[cluster_col].unique()):
        # Skip already-labeled clusters
        if cluster_id in labels and pd.notna(labels[cluster_id]):
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
        label = input("\nEnter system label (or leave blank to skip / press 'e' to edit previous): ").strip()

        if label.lower() == 'e':
            # Show previous label if any
            prev_label = labels.get(cluster_id, None)
            print(f"Previous label: {prev_label}")
            label = input("Enter new label (or leave blank to keep previous): ").strip()
            if not label:
                label = prev_label

        if label:
            labels[cluster_id] = label
        else:
            labels[cluster_id] = None

        # Save progress after each cluster
        pd.Series(labels, name='label').to_csv(progress_file)
        print(f"Progress saved to {progress_file}")

    # Apply labels to dataframe
    df[label_col] = df[cluster_col].map(labels)
    print("\n✅ Labeling complete.")
    return df, labels
