import pickle
from SentimentAnalysis.predict.classifiersPredict import SVMClassifierPredict

class Predict:
    def __init__(self, modelfilename, chisquarefilename):
        self.best_words = {}
        self.loadchisquare(chisquarefilename)
        self.svm = SVMClassifierPredict(self.best_words)
        self.svm.loadSVMModel(modelfilename)

    def predict_svm(self, predictdata):
        return self.svm.classify(predictdata)

    def loadchisquare(self, chisquarefilename):
        with open(chisquarefilename, 'rb') as f:
            self.best_words = pickle.load(f)

def createpredict(modelfilename, chisquarefilename):
    return Predict(modelfilename, chisquarefilename)

def predict(predictmodel, predictdata):
    return predictmodel.predict_svm(predictdata)

if __name__ == "__main__":
    predictmodel = createpredict("../models/svm.pickle", "../models/svm.chisquare")
    s1 = ['依然', '很', '好', '，', '实惠']
    print (s1, predict(predictmodel, s1))