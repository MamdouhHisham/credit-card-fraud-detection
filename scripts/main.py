import os
from scripts.Data_Preprocessing import load_data, scale_data, handle_imbalanced
from scripts.Model_Training import RandomForest_tuning, NeuralNet_tuning, logistic_regression_tuning
from scripts.Model_Training import RandomForest_training, NeuralNet_training, logistic_regression_training


def process_sampling_combinations(train_data):
    under = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    over = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    valid_combinations = []
    for i in under:
        for j in over:
            if i + j > 1.0:
                print(f'invalid combination: under={i}, over={j}')
                continue
            print(f'Valid combination: under={i}, over={j}')
            valid_combinations.append((i, j))
    return valid_combinations


def main():
    base_path = os.path.join("F:", "Machine learning", "Course", "2 Credit Card Fraud Detection", "data", "split")
    train_data = load_data(os.path.join(base_path, "train.csv"))
    val_data = load_data(os.path.join(base_path, "val.csv"))

    valid_combinations = process_sampling_combinations(train_data)

    X_resampled, y_resampled = handle_imbalanced(train_data, under_sample=0.4, over_sample=0.2)
    X_resampled_scaled = scale_data(X_resampled, 'minmax')

    X_val = val_data.drop(columns=['Class'])
    y_val = val_data['Class']
    X_val_scaled = scale_data(X_val, 'minmax')

    models = [RandomForest_training, NeuralNet_training, logistic_regression_training]
    for model_tn in models:
        model_tn(X_resampled_scaled, y_resampled, X_val_scaled, y_val)


if __name__ == '__main__':
    main()
