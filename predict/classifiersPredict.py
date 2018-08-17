import numpy as np
import pickle

from sklearn.svm import SVC


class SVMClassifierPredict:
    def __init__(self, best_words):
        self.clf = SVC()
        self.best_words = best_words

    def words2vector(self, all_data):
        vectors = []
        for data in all_data:
            vector = []
            for feature in self.best_words:
                vector.append(data.count(feature))
            vectors.append(vector)

        vectors = np.array(vectors)
        return vectors

    def loadSVMModel(self, filename):
        with open(filename, 'rb') as fr:
            self.clf = pickle.load(fr)

    def classify(self, data):
        vector = self.words2vector([data])
        prediction = self.clf.predict(vector)
        return prediction[0]