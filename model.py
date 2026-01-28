import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# APP TITLE & DESCRIPTION
# ---------------------------------------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.write("""
This system uses **K-Means Clustering** to group customers based on their purchasing behavior.
""")

# Load dataset
df = pd.read_csv("Wholesale customers data.csv")

# Numerical features for clustering
num_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# ---------------------------------------------------------
# SIDEBAR – USER INPUTS
# ---------------------------------------------------------
st.sidebar.header("Clustering Controls")

# Feature selections for visualization
feature1 = st.sidebar.selectbox("Select Feature 1", num_features)
feature2 = st.sidebar.selectbox("Select Feature 2", [f for f in num_features if f != feature1])

# Multi-feature selection for clustering engine
selected_features = st.sidebar.multiselect(
    "Select Features for Clustering (min 2)",
    num_features,
    default=[feature1, feature2]
)

# Guarantee Feature1 & Feature2 are included
for f in [feature1, feature2]:
    if f not in selected_features:
        selected_features.append(f)

# Validate minimum feature count
if len(selected_features) < 2:
    st.sidebar.warning("Please select at least 2 features for clustering.")

# K selection
k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)

# Random State (optional)
random_state = st.sidebar.number_input("Random State (optional)", value=42)

# Run Button
run_cluster = st.sidebar.button("🟦 Run Clustering")


# ---------------------------------------------------------
# MAIN CLUSTERING LOGIC
# ---------------------------------------------------------
if run_cluster and len(selected_features) >= 2:

    # Prepare Data
    data = df[selected_features]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Run K-Means
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(data_scaled)
    df['Cluster'] = labels

    # Cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # ---------------------------------------------------------
    # VISUALIZATION SECTION
    # ---------------------------------------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        df[feature1],
        df[feature2],
        c=df["Cluster"],
        cmap="viridis",
        s=80
    )

    # find column indices for centers
    f1_idx = selected_features.index(feature1)
    f2_idx = selected_features.index(feature2)

    # Plot cluster centers
    ax.scatter(
        centers[:, f1_idx],
        centers[:, f2_idx],
        marker='X',
        s=250,
        c='red',
        label="Cluster Centers"
    )

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.legend()
    st.pyplot(fig)

    # ---------------------------------------------------------
    # CLUSTER SUMMARY TABLE
    # ---------------------------------------------------------
    st.subheader("📘 Cluster Summary Table")

    cluster_summary = df.groupby("Cluster")[selected_features].agg(["mean", "count"])
    st.dataframe(cluster_summary)

    # ---------------------------------------------------------
    # BUSINESS INTERPRETATION
    # ---------------------------------------------------------
    st.subheader("📝 Business Interpretation")

    interpretation = ""
    global_mean = df[selected_features].mean().mean()

    for c in sorted(df["Cluster"].unique()):
        cluster_data = df[df["Cluster"] == c][selected_features]
        avg = cluster_data.mean().mean()

        if avg > global_mean:
            statement = f"🟢 **Cluster {c}: High-spending customers** – They purchase above-average across selected categories."
        elif avg < global_mean * 0.6:
            statement = f"🟡 **Cluster {c}: Budget-conscious customers** – Low overall spending, price-sensitive behavior."
        else:
            statement = f"🔵 **Cluster {c}: Moderate spenders** – Balanced purchasing habits with selective focus areas."

        interpretation += statement + "\n\n"

    st.write(interpretation)

    # ---------------------------------------------------------
    # USER GUIDANCE BOX
    # ---------------------------------------------------------
    st.info("""
    📌 **Note:** Customers in the same cluster exhibit similar purchasing behavior and 
    can be targeted with similar business strategies.
    """)

else:
    st.write("➡️ Select features and click **Run Clustering** to begin.")
