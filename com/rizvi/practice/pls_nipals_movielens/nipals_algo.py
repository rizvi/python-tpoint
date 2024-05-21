import numpy as np


# Define the NIPALS algorithm class
class NIPALS:
    def __init__(self, n_components, max_iters=100, tol=1e-4):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n, p = X.shape
        T = np.zeros((n, self.n_components))
        P = np.zeros((p, self.n_components))
        W = np.zeros((p, self.n_components))

        for i in range(self.n_components):
            t = X[:, i].copy()
            t_old = np.zeros_like(t)

            for _ in range(self.max_iters):
                # Update weights
                w = X.T @ t
                w /= np.linalg.norm(w)

                # Update scores
                t = X @ w
                t /= np.linalg.norm(t)

                # Check for convergence
                if np.linalg.norm(t - t_old) < self.tol:
                    break
                t_old = t.copy()

            # Update loadings
            p = X.T @ t / (t.T @ t)

            # Store results
            T[:, i] = t
            P[:, i] = p
            W[:, i] = w

            # Deflation
            X -= np.outer(t, p.T)

        self.T = T
        self.P = P
        self.W = W

        return self.T, self.P, self.W
