import glob
import numpy
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import LabelEncoder
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier

def conditional_mutual_info(data, x, y):
    return mutual_info_score(data[x], data[y])

def average_cmi(data):
    n_features = len(data.columns) - 1
    avg_mi = 0
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                avg_mi += conditional_mutual_info(data, data.columns[i], data.columns[j])
    return avg_mi/((n_features-1)**2)

def naive_bayes_algorithm(train_set):
    prob_table = {}
    for label in train_set['class'].unique():
        filtered = train_set[train_set['class'] == label]
        for attribute in train_set.columns[:-1]:
            for value in train_set[attribute].unique():
                class_counts = sum(1 for instance in filtered[attribute] if instance == value)
                prob_table[(label, attribute, str(value))] = class_counts/len(filtered)
    return prob_table

def logarithmic_conditional_probability(w, train_set, prob_table):
    lcp = 0
    for index, instance in train_set.iterrows():
        numerator = sum(1 for i in train_set['class'] if i == instance['class'])/len(train_set)
        for attribute, weight in zip(train_set.columns[:-1], w):
            numerator *= prob_table[(instance['class'], attribute, str(instance[attribute]))] ** weight

        denominator = 0
        for label in train_set['class'].unique():
            prob_class = sum(1 for i in train_set['class'] if i == label)/len(train_set)
            for attribute, weight in zip(train_set.columns[:-1], w):
                prob_class *= prob_table[(label, attribute, str(instance[attribute]))] ** weight
            denominator += prob_class

        lcp += numpy.log(numerator / denominator)

    return -lcp

def optimize_weights(train_set, prob_table):
    n_features = len(train_set.columns) - 1
    initial_weights = [1 for n in range(n_features)]
    bounds = [(0, 1) for n in range(n_features)]
    result = minimize(
        logarithmic_conditional_probability,
        initial_weights,
        args=(train_set, prob_table),
        bounds=bounds,
        method="L-BFGS-B",
    )
    return result.x.tolist()

def test_default_naive_bayes_algorithm(prob_table, test_set, weights):
    correct_predictions = 0
    for _, instance in test_set.iterrows():
        all_probs = []
        for value in test_set['class'].unique():
            prob_class = sum(1 for i in test_set['class'] if i == value)/len(test_set)
            for index, attribute in enumerate(test_set.columns[:-1]):
                key=(value, attribute, str(instance[attribute]))
                prob_class *= prob_table[key]
            all_probs.append(prob_class)
        predicted_class = test_set['class'].unique()[all_probs.index(max(all_probs))]
        if predicted_class == instance['class']:
            correct_predictions += 1
    return correct_predictions/len(test_set)

def test_weighted_naive_bayes_algorithm(prob_table, test_set, weights):
    correct_predictions = 0
    for _, instance in test_set.iterrows():
        all_probs = []
        for value in test_set['class'].unique():
            prob_class = sum(1 for i in test_set['class'] if i == value)/len(test_set)
            for index, attribute in enumerate(test_set.columns[:-1]):
                key=(value, attribute, str(instance[attribute]))
                prob_class *= prob_table[key] ** weights[index]
            all_probs.append(prob_class)
        predicted_class = test_set['class'].unique()[all_probs.index(max(all_probs))]
        if predicted_class == instance['class']:
            correct_predictions += 1
    return correct_predictions/len(test_set)

def main():
    files = sorted(glob.glob('data/*'))
    for f in range(0, len(files), 3):
        print()
        data = pd.read_csv(files[f])
        test_set = pd.read_csv(files[f+1])
        train_set = pd.read_csv(files[f+2])
        print("Dataset:", files[f].split('/')[1])
        average_cmi_value = average_cmi(data)
        print("Average Conditional Mutual Information:", f"{round(average_cmi_value, 2)}")
        prob_table = naive_bayes_algorithm(train_set)
        weights = optimize_weights(train_set, prob_table)
        #print("Weights:", weights)
        default_nb_accuracy = test_default_naive_bayes_algorithm(prob_table, test_set, weights)
        print("Default Naive Bayes Classifier Accuracy:", f"{round(default_nb_accuracy*100, 2)}%")
        weighted_nb_accuracy = test_weighted_naive_bayes_algorithm(prob_table, test_set, weights)
        print("Weighted Naive Bayes Classifier Accuracy:", f"{round(weighted_nb_accuracy*100, 2)}%")

        X_train = train_set.iloc[:, :-1]
        y_train = train_set.iloc[:, -1]
        X_test = test_set.iloc[:, :-1]
        y_test = test_set.iloc[:, -1]

        pd.options.mode.copy_on_write = True
        label_encoders = {}
        for column in X_train.columns:
            le = LabelEncoder()
            X_train[column] = le.fit_transform(X_train[column])
            X_test[column] = le.transform(X_test[column])
            label_encoders[column] = le
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(y_train)
        y_test = target_encoder.transform(y_test)

        dt_classifier = DecisionTreeClassifier(random_state=1)
        dt_classifier.fit(X_train, y_train)
        y_pred_dt = dt_classifier.predict(X_test)
        dt_accuracy = accuracy_score(y_test, y_pred_dt)
        print("Decision Tree Classifier Accuracy:", f"{round(dt_accuracy*100, 2)}%")

if __name__ == '__main__':
    main()