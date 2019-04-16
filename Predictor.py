import pandas as pd
from sklearn.externals import joblib
import json


class Predictor:

    def __init__(self, filename_model, filename_features):

        self.model, self.features = Predictor.load_model(filename_model, filename_features)
        self.data = None

    @staticmethod
    def load_model(filename_model, filename_features):
        """
        Loading saved fitted model and used features from files
        :param filename_model: fitted model
        :param filename_features: list of used features
        :return: loaded_model, loaded_features
        """

        loaded_model = joblib.load(filename_model)
        loaded_features = joblib.load(filename_features)

        return loaded_model, loaded_features

    def load_data(self, filename = None, json_str = None):
        """
        Loading data from JSON file for making prediction
        :param filename: JSON filename
        :param json_str: JSON-string
        """
        if (filename is None) and (json_str is None):
            raise ValueError("Filename or json must be specified!")

        if json_str is not None:
            data = json.loads(json_str)
            data = pd.DataFrame(data).T
        else:
            data = pd.read_json(filename, orient='index')

        if (set(self.features).issubset(set(data.columns))) == False:
            raise ValueError("Data doesn't contain all necessary features!")

        data = data[self.features]

        data = data[data.columns].apply(pd.to_numeric, errors='ignore')

        self.data = data

        return

    def predict(self):
        """
        Making predictions
        :return: JSON object
        """
        predictions = self.model.predict(self.data)
        predictions = zip(list(self.data.index), predictions )

        l = [(str(v1), str(v2)) for (v1, v2) in predictions]

        return json.dumps(dict(l))

