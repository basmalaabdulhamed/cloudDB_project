from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def run_clustering(df, k=4):

    features = [
        "MonthlyIncome",
        "JobSatisfaction",
        "PerformanceRating",
        "YearsAtCompany",
        "EngagementScore"
    ]

    X = df[features]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    # Elbow Method
    inertia_values = []
    K_range = range(1, 11)

    for i in K_range:
        elbow_model = KMeans(n_clusters=i, random_state=42)
        elbow_model.fit(x_scaled)
        inertia_values.append(elbow_model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia_values, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

    # Final KMeans model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )
    df["Cluster"] = kmeans.fit_predict(x_scaled)

    # Silhouette Score
    score = silhouette_score(
    x_scaled,
    df["Cluster"]
    )

    print("\nSilhouette Score:", round(score, 3))

    # pca visualization
    pca = PCA(n_components=2)

    pca_components = pca.fit_transform(x_scaled)

    df["PCA1"] = pca_components[:, 0]
    df["PCA2"] = pca_components[:, 1]

    plt.figure(figsize=(10, 7))

    plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["Cluster"],
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.title("Employee Clusters Visualization")

    plt.show()

    return df, kmeans