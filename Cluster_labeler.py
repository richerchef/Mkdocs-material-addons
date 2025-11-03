from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import os

def cluster_and_label_systems(
    df,
    text_col='title_clean',
    system_col='system',
    n_clusters=10,
    progress_file='cluster_label_progress.csv'
):
    """
    Cluster rows with missing system values, show examples, and let the user
    assign existing system labels interactively.

    - Only processes rows where system is NaN
    - Clusters on text_col
    - Suggests existing system labels from df
    - Saves progress incrementally
    """

    # Work only on rows missing system
    df_target = df[df[system_col].isna()].copy()
    print(f"🧭 Found {len(df_target)} rows with missing system labels")

    # If nothing to label, stop early
    if len(df_target) == 0:
        print("✅ No missing system labels found.")
        return df

    # TF-IDF vectorization
    print("🔍 Vectorizing text...")
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df_target[text_col])

    # Dynamic cluster suggestion (optional)
    if n_clusters == "auto":
        n_clusters = max(2, min(20, len(df_target) // 50))

    print(f"🌀 Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_target['cluster'] = kmeans.fit_predict(X)

    # Load previous progress if any
    if os.path.exists(progress_file):
        progress = pd.read_csv(progress_file)
        cluster_map = dict(zip(progress['cluster'], progress['system_label']))
        print(f"Resuming from progress file: {progress_file}")
    else:
        cluster_map = {}

    existing_systems = sorted(df[system_col].dropna().unique().tolist())

    for cluster_id in sorted(df_target['cluster'].unique()):
        if cluster_id in cluster_map:
            continue  # Skip already labeled clusters

        cluster_rows = df_target[df_target['cluster'] == cluster_id][text_col].tolist()
        print("\n" + "="*70)
        print(f"🧩 Cluster {cluster_id} — {len(cluster_rows)} items")
        print("-"*70)

        for i, row_text in enumerate(cluster_rows, 1):
            print(f"{i:>3}. {row_text}")

        print("\n💡 Existing system options:")
        print(", ".join(existing_systems))
        print("-"*70)

        label = input("Type system name (or leave blank to skip): ").strip()

        if label:
            cluster_map[cluster_id] = label
            df_target.loc[df_target['cluster'] == cluster_id, system_col] = label
        else:
            cluster_map[cluster_id] = None

        # Save progress
        pd.DataFrame({'cluster': list(cluster_map.keys()), 'system_label': list(cluster_map.values())}).to_csv(progress_file, index=False)
        print(f"✅ Progress saved ({progress_file})")

    print("\n🎯 Updating main DataFrame...")
    df.update(df_target)
    print("✅ System labels updated successfully.")
    return df
