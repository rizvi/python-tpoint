import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier


def pretreat(X, method):
    if method == 1:  # Autoscaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif method == 2:  # Centering
        X_scaled = X - np.mean(X, axis=0)
    elif method == 3:  # Unilength
        norms = np.linalg.norm(X, axis=0)
        X_scaled = X / norms
    elif method == 4:  # Min-Max scaling
        X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 5:  # Pareto scaling
        X_scaled = (X - np.mean(X, axis=0)) / np.sqrt(np.std(X, axis=0))
    else:
        print('Wrong data pretreat method!')
        return None

    return X_scaled


def pls_nipals(X, Y, n_components):
    T = np.zeros((X.shape[0], n_components))
    P = np.zeros((X.shape[1], n_components))
    W = np.zeros((X.shape[1], n_components))
    U = np.zeros((Y.shape[0], n_components))
    Q = np.zeros((n_components, Y.shape[1]))

    X_copy = X.copy()
    Y_copy = Y.copy()

    for i in range(n_components):
        u = Y_copy[:, 0].copy()  # Initial guess for u
        t = np.zeros(X.shape[0])
        w = np.zeros(X.shape[1])
        q = np.zeros(Y.shape[1])

        # Iterative process
        for _ in range(100):  # Limiting the number of iterations to ensure convergence
            w = np.dot(X_copy.T, u) / np.linalg.norm(np.dot(X_copy.T, u))
            t = np.dot(X_copy, w)
            q = np.dot(Y_copy.T, t) / np.dot(t.T, t)
            u_new = np.dot(Y_copy, q) / np.dot(q.T, q)

            if np.linalg.norm(u_new - u) < 1e-10:  # Convergence check
                break
            u = u_new

        # Deflate X and Y
        p = np.dot(X_copy.T, t) / np.dot(t.T, t)
        X_copy -= np.outer(t, p.T)
        Y_copy -= np.outer(t, q.T)

        # Store the components
        T[:, i] = t
        P[:, i] = p
        W[:, i] = w
        Q[i, :] = q
        U[:, i] = u

    # Calculate the regression coefficient matrix B
    B = np.dot(np.dot(W, np.linalg.inv(np.dot(P.T, W).T)), Q)

    return B, W, P, T, U, Q


def plot_regression_results(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, edgecolor='k', alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.show()


# Load your dataset
dataset_path = r'E:\MIT-IIT-DU\3rd_Semester\Dr. Mohammad Shoyaib Sir\Dataset\sonar.csv'
df = pd.read_csv(dataset_path)
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values  # Target

# Convert target variable to numeric if it's not
le = LabelEncoder()
y = le.fit_transform(y).astype(float)  # Convert to float

# Choose your method for data pretreatment
print("Choose data pretreatment method:")
print("1. Autoscaling")
print("2. Centering")
print("3. Unilength")
print("4. Min-Max Scaling")
print("5. Pareto Scaling")
method = int(input("Enter the method number (1-5): "))

# Pretreat the data
X_scaled = pretreat(X, method)

# Check if the pretreatment was successful
if X_scaled is not None:
    mse_pls_list = []
    r2_pls_list = []
    accuracy_knn_list = []
    f1_knn_list = []
    recall_knn_list = []
    precision_knn_list = []

    for i in range(10):
        # Split the dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=None)

        # Perform PLS using NIPALS algorithm
        n_components = 2  # Define the number of PLS components
        B_pls, W_pls, P_pls, T_pls, U_pls, Q_pls = pls_nipals(X_train, y_train.reshape(-1, 1), n_components)

        # Perform prediction on the test set using PLS
        y_pred_pls = np.dot(X_test, B_pls)

        # Calculate the metrics for PLS
        mse_pls = mean_squared_error(y_test, y_pred_pls)
        r2_pls = r2_score(y_test, y_pred_pls)
        mse_pls_list.append(mse_pls)
        r2_pls_list.append(r2_pls)

        # Apply k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        # Calculate the metrics for k-NN
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
        recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
        precision_knn = precision_score(y_test, y_pred_knn, average='weighted')

        accuracy_knn_list.append(accuracy_knn)
        f1_knn_list.append(f1_knn)
        recall_knn_list.append(recall_knn)
        precision_knn_list.append(precision_knn)

        # Plot the regression results
        plot_regression_results(y_test, y_pred_pls, f'Regression Results Before Classification (Iteration {i + 1})')

    # Print the coefficient matrix
    print("Coefficient matrix B:")
    print(B_pls)

    # Calculate and print the average metrics
    avg_mse_pls = np.mean(mse_pls_list)
    avg_r2_pls = np.mean(r2_pls_list)
    avg_accuracy_knn = np.mean(accuracy_knn_list)
    avg_f1_knn = np.mean(f1_knn_list)
    avg_recall_knn = np.mean(recall_knn_list)
    avg_precision_knn = np.mean(precision_knn_list)

    print("Average Results for PLS over 10 iterations:")
    print(f'Average Mean Squared Error: {avg_mse_pls}')
    print(f'Average R2 Score: {avg_r2_pls}')

    print("Average Results for k-NN Classifier over 10 iterations:")
    print(f'Average Accuracy: {avg_accuracy_knn}')
    print(f'Average F1 Score: {avg_f1_knn}')
    print(f'Average Recall: {avg_recall_knn}')
    print(f'Average Precision: {avg_precision_knn}')
else:
    print("Data pretreatment failed due to an incorrect method choice.")