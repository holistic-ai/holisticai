import jax.numpy as jnp
from jax import vmap


class KNeighborsClassifier:
    def fit(self, X_train, y_train):
        self.X_train = jnp.array(X_train)
        self.y_train = jnp.array(y_train)
        self.n_train_samples = len(X_train)

    def kneighbors_batched(self, X, n_neighbors=None, return_distance=True, batch_size=100):
        if n_neighbors is None:
            n_neighbors = self.n_train_samples

        n_samples = X.shape[0]

        for i in range(0, n_samples, batch_size):
            X_batch = jnp.array(X[i : i + batch_size])
            dists_batch = jnp.sqrt(jnp.sum((X_batch[:, None, :] - self.X_train[None, :, :]) ** 2, axis=-1))
            batch_neighbors_idx = jnp.argsort(dists_batch, axis=1)[:, :n_neighbors]
            if return_distance:
                yield dists_batch, batch_neighbors_idx
            else:
                yield batch_neighbors_idx

    def kneighbors(self, X, n_neighbors=None, return_distance=True, batch_size=100):
        neighbors_dists = []
        neighbors_idx = []

        for dists_batch, batch_neighbors_idx in self.kneighbors_batched(X, n_neighbors, return_distance, batch_size):
            batch_neighbors_dists = jnp.take_along_axis(dists_batch, batch_neighbors_idx, axis=1)
            neighbors_dists.append(batch_neighbors_dists)
            neighbors_idx.append(batch_neighbors_idx)

        neighbors_idx = jnp.vstack(neighbors_idx)
        if return_distance:
            neighbors_dists = jnp.vstack(neighbors_dists)
            return neighbors_dists, neighbors_idx
        return neighbors_idx

    def _predict_single(self, x):
        # Compute distances from x to all training points
        dists = jnp.sqrt(jnp.sum((self.X_train - x) ** 2, axis=-1))
        # Find the k nearest neighbors
        neighbors = jnp.argsort(dists)[: self.n_neighbors]
        # Predict the label by majority vote
        unique, counts = jnp.unique(self.y_train[neighbors], return_counts=True)
        return unique[jnp.argmax(counts)]

    def predict(self, X):
        # Vectorize prediction for all input points
        predictions = vmap(self._predict_single)(jnp.array(X))
        return predictions
