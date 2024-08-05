from lib.Experimental import prepare_dataset, prepare_extended_dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from lib.Utils import Utils
import logging
import keras
import tensorflow as tf
import pickle
import os
from sklearn.dummy import DummyClassifier

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
np.set_printoptions(threshold=np.inf)

def getCNN(num_classes, input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters= 32, kernel_size=3, activation='relu',padding='same',input_shape= input_shape[2:]))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3,padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3,padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    model.summary()
    return model

selected_metrics = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
}

n_splits=5
n_repeats=1 # as StratifiedKFold is used
skf = StratifiedKFold(n_splits=n_splits, random_state=1410, shuffle=True)
feature_window_size = 12

# if __name__ == "__main__":
#     # X, Y = prepare_dataset(feature_window_size=feature_window_size)
#     # input_shape = (1, 1, 12, 1)
#     # filename = "raw"

#     X, Y = prepare_extended_dataset(
#         included_fields=[2,3], # include only metrics
#         feature_window_size=feature_window_size)
#     input_shape = (1, 1, 24, 1)
#     filename = "extended"

#     unique_labels = set(np.unique(Y))

#     logger.info(f"I'm starting the script. Dataset is loaded. Dataset size={X.shape[0]}, labels={set(np.unique(Y))}")

    # selected_classifiers = {
    #     "CNN": getCNN(num_classes = len(unique_labels), input_shape=input_shape),
    #     "RFC_20": RandomForestClassifier(max_depth=20, random_state=0),
    #     "DTC": DecisionTreeClassifier(random_state=0)
    # }

#     all_results = {classifier: {metric: [] for metric in selected_metrics} for classifier in selected_classifiers}

#     for i, (train, test) in enumerate(skf.split(X, Y)):
#         logger.info(f"Iteration {i} started.")
#         for classifier in selected_classifiers:
#             X_train, y_train, X_test, y_test = X[train], Y[train], X[test], Y[test]
#             logger.info(f"    Classifier={classifier}.")
#             clf = selected_classifiers[classifier]

#             if "CNN" in classifier:
#                 X_train = X_train[..., np.newaxis]
#                 X_test = X_test[..., np.newaxis]

#             clf.fit(X_train, y_train.ravel())

#             y_pred = clf.predict(X_test)
#             if "CNN" in classifier:
#                 y_pred = np.apply_along_axis(lambda a: np.argmax(a), 1, y_pred)

#             Utils.save_confusion_matrix(y_test, y_pred, unique_labels, experimentName=f"{classifier}_{i}")

#             for metric in selected_metrics:
#                 result = selected_metrics[metric](y_test.ravel(), y_pred)
#                 all_results[classifier][metric].append(result)
#                 logger.info(f"    Classifier passed. Results: {result}")

#             logger.info(f"Iteration {i} passed.")

#     print(all_results)
#     Utils.print_aggregated_cv_results(all_results)

#     clfs_len = len(selected_classifiers)
#     for selected_metric in selected_metrics:
#         print(selected_metric)
#         scores = np.zeros((clfs_len, n_splits * n_repeats))

#         for idx, classifier in enumerate(all_results):
#             for metric in all_results[classifier]:
#                 if metric == selected_metric:
#                     scores[idx] = all_results[classifier][metric]

#         Utils.compare_classifiers(scores, clfs_len, selected_classifiers)
#         print("-"*70)

#     with open(f'{filename}.pkl', 'wb') as f:
#         pickle.dump(all_results, f)

##################################################

aggregated_dict = {}

def read_dict(filename):
    with open(f'{filename}.pkl', 'rb') as fp:
        dict = pickle.load(fp)
        # print(dict)
        return dict


DIRNAME = "." 
raw = read_dict(os.path.join(DIRNAME, 'raw'))

extended = read_dict(os.path.join(DIRNAME, 'extended'))
for old_key in extended:
    new_key = old_key+"_e"
    extended[new_key] = extended.pop(old_key)

# aggregated_dict = extended | raw
aggregated_dict = dict(list(extended.items()) + list(raw.items()))

selected_classifiers = {
    "CNN": DummyClassifier(),
    "RFC_20": DummyClassifier(),
    "DTC": DummyClassifier(),
    "CNN_e": DummyClassifier(),
    "RFC_20_e": DummyClassifier(),
    "DTC_e": DummyClassifier(),
}

# print(aggregated_dict)

clfs_len = 6
for selected_metric in selected_metrics:
    print(selected_metric)
    scores = np.zeros((clfs_len, n_splits * n_repeats))

    for idx, classifier in enumerate(aggregated_dict):
        for metric in aggregated_dict[classifier]:
            if metric == selected_metric:
                scores[idx] = aggregated_dict[classifier][metric]

    Utils.compare_classifiers(scores, clfs_len, selected_classifiers)
    print("-"*70)
