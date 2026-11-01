import numpy as np

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)

    for _ in range(max_iterations):
        # 1. Compute Euclidean distances between all points and centroids
        # Shape: (num_points, num_centroids)
        distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)

        # 2. Assign points to nearest centroid
        cluster_assignments = np.argmin(distances, axis=1)

        # 3. Update centroids
        new_centroids = np.copy(centroids)
        for i in range(k):
            cluster_points = points[cluster_assignments == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)

        # Stop early if centroids didn't move
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # 4. Round coordinates to 4 decimal places and format output as tuples
    final_centroids = [tuple(np.round(c, 4)) for c in centroids]
    return final_centroids