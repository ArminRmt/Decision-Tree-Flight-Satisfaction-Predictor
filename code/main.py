import pandas as pd
import numpy as np
from decision_tree import DecisionTree
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_FOLDER = "./../results"
DATA_FILE_PATH = "./flight_satisfaction.csv"


def load_data(file_path):
    return pd.read_csv(file_path)


def save_result_to_file(result, output_file):
    with open(output_file, "w") as f:
        f.write(result)


def preprocess_data(data):
    X = data.drop(["satisfaction", "id"], axis=1)
    y = np.where(data["satisfaction"] == "satisfied", 1, 0)  # binary classification

    # X = handle_missing_values(X)
    # X = scale_numerical_features(X)

    return X, y


def handle_missing_values(X):
    columns_with_missing = X.columns[X.isnull().any()]

    if len(columns_with_missing) > 0:
        for col in columns_with_missing:
            if np.issubdtype(X[col].dtype, np.floating):
                X[col].fillna(X[col].mean(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True)

    return X


def scale_numerical_features(X):
    numerical_columns = X.select_dtypes(include=[np.number]).columns

    for col in numerical_columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    return X


def train_decision_tree(X_train, y_train, max_depth=5):
    clf = DecisionTree(max_depth=max_depth)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def visualize_correlation_matrix(X, rows, output_folder=RESULTS_FOLDER):
    numerical_columns = X.select_dtypes(include=[np.number]).columns
    corr_matrix = X[numerical_columns].corr()

    plt.figure(figsize=(14, 14))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    rows_folder = os.path.join(output_folder, f"rows_{rows}")
    if not os.path.exists(rows_folder):
        os.makedirs(rows_folder)

    heatmap_filename = os.path.join(rows_folder, "correlation_matrix_heatmap.png")

    if not os.path.exists(heatmap_filename):
        plt.savefig(heatmap_filename)
        plt.close()
        return True
    else:
        plt.close()
        return False


def save_distribution_plots(
    X, rows, subfolder="distribution_plots", output_folder=RESULTS_FOLDER
):
    subfolder_path = os.path.join(output_folder, f"rows_{rows}", subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    plots_saved = False
    for column in X.columns:
        sanitized_column_name = column.replace("/", "_").replace(" ", "_")
        plot_filename = os.path.join(
            subfolder_path, f"distribution_of_{sanitized_column_name}.png"
        )

        if not os.path.exists(plot_filename):
            plt.figure(figsize=(12, 6))
            sns.histplot(X[column], kde=True, color="blue")
            plt.title(f"Distribution of {column}")
            plt.savefig(plot_filename)
            plt.close()
            plots_saved = True

    if plots_saved:
        print(f"Distribution plots saved to '{subfolder_path}'")
    return plots_saved


def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[indices[:split_index]], X.iloc[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    return X_train, X_test, y_train, y_test


def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy


def calculate_and_save_information_gain(
    clf, X, y, rows, top_n=10, output_folder=RESULTS_FOLDER
):
    output_file_name = "information_gain_results.csv"
    information_gains = clf.calculate_information_gain_for_all_features(X, y)

    sorted_information_gains = dict(
        sorted(information_gains.items(), key=lambda item: item[1], reverse=True)
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    rows_folder = os.path.join(output_folder, f"rows_{rows}")
    if not os.path.exists(rows_folder):
        os.makedirs(rows_folder)

    output_path = os.path.join(rows_folder, output_file_name)

    with open(output_path, "w") as file:
        file.write("Feature,Information Gain\n")
        for feature, gain in list(sorted_information_gains.items())[:top_n]:
            file.write(f"{feature},{gain}\n")

    features_to_plot = list(sorted_information_gains.keys())[:top_n]
    gains_to_plot = list(sorted_information_gains.values())[:top_n]

    plt.figure(figsize=(12, 12))
    plt.bar(features_to_plot, gains_to_plot, color="skyblue")
    plt.ylabel("Information Gain")
    plt.title(f"Top {top_n} Features: Information Gain")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")

    figure_path = os.path.join(rows_folder, "information_gain_plot.png")
    plt.savefig(figure_path)

    print(f"Information Gain results (top {top_n} features) saved to '{output_path}'")
    print(f"Figure saved to '{figure_path}'")


def run_experiment(rows, random_state=None):
    data = load_data(os.path.join(os.path.dirname(__file__), DATA_FILE_PATH))

    data_subset = data.sample(rows, random_state=random_state)

    X, y = preprocess_data(data_subset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    clf = train_decision_tree(X_train, y_train)

    info_gain_saved = calculate_and_save_information_gain(clf, X_train, y_train, rows)

    accuracy = evaluate_model(clf, X_test, y_test)
    print(f"Accuracy for {rows} rows: {accuracy}")

    visualize_saved = visualize_correlation_matrix(X, rows)
    distribution_plots_saved = save_distribution_plots(X, rows)

    if visualize_saved or info_gain_saved or distribution_plots_saved:
        print("Some results were saved.")


if __name__ == "__main__":
    run_experiment(rows=2000, random_state=42)
    run_experiment(rows=5000, random_state=42)
