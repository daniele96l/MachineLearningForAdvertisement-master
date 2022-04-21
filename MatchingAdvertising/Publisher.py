import numpy as np
from hungarian_algorithm import hungarian_algorithm as hungarian_algorithm
from hungarian_algorithm import convert_matrix as convert_matrix


class Publisher:
    def __init__(self, n_slots):
        if n_slots <= 3:
            raise SystemExit("Number of slots should be greater than 3")
        self.n_slots = n_slots
        self.slots = np.array([[] for i in range(n_slots)])  # or just [] or np.array[]

    def allocate_ads(self, samples):
        n_ads = len(samples)
        graph_matrix = np.zeros(shape=(n_ads, self.n_slots))

        for i in range(n_ads):
            for j in range(self.n_slots):
                graph_matrix[i][j] = samples[i][j]
        # print("Ads allocating:")
        # print(graph_matrix)
        # print("Running hungarian...")
        res = hungarian_algorithm(convert_matrix(graph_matrix))
        m = res[1]
        edges = []
        for j in range(self.n_slots):
            for i in range(n_ads):
                if m[i][j] == 1:
                    edges.append([i, j])
        # print(edges)
        return edges
