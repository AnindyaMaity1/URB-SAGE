import numpy as np


def build_adjacency(zones, neighbors=3):
    """Connect each zone to its nearest peers and return normalized adjacency."""
    coordinates = np.array([[zone["latitude"], zone["longitude"]] for zone in zones], dtype=np.float32)
    distances = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2))
    adjacency = np.zeros_like(distances)
    for index in range(len(zones)):
        nearest = np.argsort(distances[index])[1:neighbors + 1]
        adjacency[index, nearest] = 1
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(len(zones), dtype=np.float32)
    degree = adjacency.sum(axis=1)
    return adjacency / np.sqrt(degree[:, None] * degree[None, :])