import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.externals import joblib

from Preprocessor import Preprocessor


class TrainModel:

    @staticmethod
    def load_data(path):
        """
        Method for loading train data from xlsx file
        :param path: path to the input file with train data
        :return: pandas dataframe
        """
        # path = 'data/data.xlsx'
        data_xls = pd.read_excel(path)

        # All headers are in one cell. Splitting and refining them
        # Data in rows need to be splitted too
        cols_list = list(data_xls.columns)[0].split(',')
        cols_list = [c.replace('"','').replace(' ','_') for c in cols_list]
        data = pd.DataFrame(data_xls[data_xls.columns[0]].str.split(',').tolist(), columns = cols_list[:-1])

        # Diagnosis is binary. Replacing symbolic values by 0 and 1
        data['diagnosis'] = data['diagnosis'].map({'B':'0','M':'1'})
        data = data[data.columns].apply(pd.to_numeric, errors='ignore')

        # Check id's to be unique
        if (data['id'].nunique() != data.shape[0]):
            print('Warning! Id values are not unique!')

        data.set_index('id', inplace=True)

        print("Input data shape: ", data.shape)

        return data

    def get_train_test(self, data, features, target, random_state, test_size=0.3):
        """
        Splits dataset into train and holdout sets
        :param data: whole dataset, pandas dataframe
        :param features: list of features
        :param target: name of target feature, str
        :param random_state: random state for reproducibility, int
        :param test_size: fraction of data for holdout set, float
        :return: train_data, test_data, train_labels, test_labels
        """
        # Get train and holdout sets
        train_data, test_data, train_labels, test_labels = train_test_split(data[features], data[target],
                                                                            test_size=test_size,
                                                                            random_state=random_state)
        return train_data, test_data, train_labels, test_labels

    def search_best_knn(self, data, features, target, random_state, cv_splits=5, max_neighbours=25, scoring='recall', test_size=0.3):
        """
        Searching best estimator using GridSearchCV. Uses pipeline.
        :param data: dataset (pandas dataframe)
        :param features: features to use from dataset, list
        :param target: name of target feature
        :param random_state: random state for reproducibility
        :param cv_splits: number of splits for cross-validation
        :param max_neighbours: max number of neighbours to use in knn
        :param scoring: scoring method (one of 'accuracy', 'f1', 'f1_weighted', 'precision',
                                        'recall', 'recall_weighted', 'roc_auc'), str
        :param test_size: test_size: fraction of data for holdout set, float
        :return: best_estimator pipeline
        """

        steps = [('preprocessor', Preprocessor()),
                 ('scaler', StandardScaler()),
                 ('knn', KNeighborsClassifier(algorithm='brute', metric='minkowski'))]
        pipeline = Pipeline(steps)

        train_data, test_data, train_labels, test_labels = self.get_train_test(data, features,
                                                                               target, random_state,
                                                                               test_size=test_size)

        if len(train_data) < 2:
            raise ValueError("Train data size is too small")

        if scoring not in ('accuracy', 'f1', 'f1_weighted', 'precision', 'recall', 'recall_weighted', 'roc_auc'):
            raise ValueError("This scoring method is not supported")

        # We'll be choose optimal neighbours number, but not greater than max_neighbours
        # Also neighbours number depends on input data size
        up_lim = int(np.floor(len(train_data) * (cv_splits - 1) / cv_splits))
        if up_lim > max_neighbours:
            up_lim = max_neighbours

        parameters_grid = {
            'knn__weights': ['uniform', 'distance'],
            'knn__n_neighbors': range(1, up_lim),
            'knn__p': range(1, 4)}

        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        grid_cv = GridSearchCV(pipeline, param_grid=parameters_grid, cv=skf, scoring=scoring, n_jobs=-1)
        grid_cv.fit(train_data, train_labels)

        best_score = grid_cv.best_score_
        best_estimator = grid_cv.best_estimator_
        print("Best estimator: ", best_estimator)
        print("Best CV score ", best_score)
        print("")

        best_estimator.fit(train_data, train_labels)

        self.evaluate_model(best_estimator, test_data, test_labels, metrics=['accuracy', 'precision', 'recall', 'f1'])

        return best_estimator

    def evaluate_model(self, estimator, test_data, test_labels, metrics=[]):
        """
        Evaluates metrics for estimator on the provided data
        :param estimator: esimator to evaluate
        :param test_data: test_data
        :param test_labels: test_labels
        :param metrics: list of metrics to evaluate
        :return:
        """

        if not metrics:
            return

        estimator_pred = estimator.predict(test_data)

        metrics_methods = {
            'accuracy': accuracy_score(test_labels, estimator_pred),
            'precision': precision_score(test_labels, estimator_pred),
            'recall': recall_score(test_labels, estimator_pred),
            'f1': f1_score(test_labels, estimator_pred)
        }

        for metric in metrics:
            print(metric + ' score: ', metrics_methods.get(metric, 'not supported'))

        return

    def save_model(self, best_estimator, features, filename_prefix):
        """
        Saves fitted model and used features to files
        :param best_estimator: fitted estimator to save
        :param features: features used
        :param filename_prefix:
        :return: filename_model, filename_features
        """

        filename_model = filename_prefix + '_model.sav'
        filename_features = filename_prefix + '_features.sav'

        joblib.dump(best_estimator, filename_model)
        joblib.dump(features, filename_features)

        return filename_model, filename_features



