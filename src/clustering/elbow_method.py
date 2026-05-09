from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def plot_elbow_method(X):

    inertia_values = []

    K_values = range(2, 11)

    for k in K_values:

        model = KMeans(
            n_clusters=k,
            random_state=42
        )

        model.fit(X)

        inertia_values.append(model.inertia_)

    plt.figure(figsize=(8, 5))

    plt.plot(
        K_values,
        inertia_values,
        marker="o"
    )

    plt.title("Elbow Method")

    plt.xlabel("Number of Clusters (K)")

    plt.ylabel("Inertia")

    plt.grid(True)

    plt.savefig(
        "Reports/charts/elbow_method.png"
    )

    plt.show()