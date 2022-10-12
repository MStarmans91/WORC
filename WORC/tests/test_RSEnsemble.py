from WORC.classification.crossval import test_RS_Ensemble
import pandas as pd
import os

classification_data = r"C:\Users\Martijn Starmans\Documents\GitHub\WORCTutorial\WORC_Example_STWStrategyHN_220915_DoTstNRSNEns\classify\all\tempsave\tempsave_0.hdf5"

# Read the data and take first predicted label
classification_data = pd.read_hdf(classification_data)
classification_data = classification_data[classification_data.keys()[0]]

# print(classification_data)

# Iterate over cross-validation iterations
# for i_cv in range(len(classification_data.classifiers)):
#     trained_classifier = classification_data.classifiers[i_cv]
#     X_train = classification_data.X_train[i_cv]
#     Y_train = classification_data.Y_train[i_cv]
#     X_test = classification_data.X_test[i_cv]
#     Y_test = classification_data.Y_test[i_cv]
#     feature_labels = classification_data.feature_labels
# output_json = os.path.join(os.path.dirname(__file__), f'performance_{i_cv}.json')

# For tempsave
trained_classifier = classification_data.trained_classifier
trained_classifier.fitted_validation_workflows = list()
X_train = classification_data.X_train
Y_train = classification_data.Y_train
X_test = classification_data.X_test
Y_test = classification_data.Y_test
feature_labels = classification_data.feature_labels
output_json = os.path.join(os.path.dirname(__file__), 'performance_0.json')

# print(trained_classifier.fitted_workflows)
# print(trained_classifier.fitted_validation_workflows)

test_RS_Ensemble(estimator_input=trained_classifier,
                    X_train=X_train, Y_train=Y_train,
                    X_test=X_test, Y_test=Y_test,
                    feature_labels=feature_labels,
                    output_json=output_json,
                    verbose=True)