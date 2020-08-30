# 데이터의 전처리 및 분할을 수행한다.
import joblib
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class Preprocessor():
    def preprocess(self, path="dataset", xydata=None, test_size=0.2):
        # 전처리한다.

        # 1. raw 데이터 가져오기
        if not xydata:
            xdata, ydata = joblib.load(path)
        else:
            xdata, ydata = xydata

        # 2. 데이터 섞기
        xdata, ydata = shuffle(xdata, ydata)

        # 3. 라벨 형식 설정
        labels = [[1, 0], [0, 1]]
        ydata = np.array([labels[i] for i in ydata])

        # 4. train test 데이터 split
        xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=test_size)

        return xtrain, xtest, ytrain, ytest

    def get_count(self, ydata, label):
        # ydata의 label 개수를 반환한다.
        return (ydata == label).sum()
