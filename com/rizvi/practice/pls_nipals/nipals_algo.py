import numpy as np


# Define the NIPALS algorithm class
class NIPALS:
    def __init__(self, n_components, max_iters=100, tol=1e-4):
        # Initialize the number of components, maximum iterations, and tolerance for convergence
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n, p = X.shape  # Get the number of samples (n) and features (p)
        T = np.zeros((n, self.n_components))  # Initialize the score matrix
        P = np.zeros((p, self.n_components))  # Initialize the loading matrix
        W = np.zeros((p, self.n_components))  # Initialize the weight matrix

        for i in range(self.n_components):
            t = X[:, i].copy()  # Initialize the score vector with the i-th column of X
            t_old = np.zeros_like(t)  # Initialize the old score vector for convergence checking

            for _ in range(self.max_iters):
                # Update weights
                w = X.T @ t  # Calculate the weight vector
                w /= np.linalg.norm(w)  # Normalize the weight vector

                # Update scores
                t = X @ w  # Calculate the new score vector
                t /= np.linalg.norm(t)  # Normalize the score vector

                # Check for convergence
                if np.linalg.norm(t - t_old) < self.tol:
                    break  # Exit the loop if the score vector has converged
                t_old = t.copy()  # Update the old score vector

            # Update loadings
            p = X.T @ t / (t.T @ t)  # Calculate the loading vector

            # Store results
            T[:, i] = t  # Store the score vector in the score matrix
            P[:, i] = p  # Store the loading vector in the loading matrix
            W[:, i] = w  # Store the weight vector in the weight matrix

            # Deflation
            X -= np.outer(t, p.T)  # Subtract the outer product of the score and loading vectors from X

        self.T = T  # Store the final score matrix
        self.P = P  # Store the final loading matrix
        self.W = W  # Store the final weight matrix

        return self.T, self.P, self.W
