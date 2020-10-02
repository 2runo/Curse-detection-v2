# 모델을 학습한다.
import models
import preprocessing
import numpy as np
import joblib


# 데이터 전처리 및 분할
pre = preprocessing.Preprocessor()

try:
    mfcc_x, mfcc_y, ft_x, ft_y = joblib.load('dataset/data.joblib')
except:
    raise Exception('embedding.py를 먼저 실행해 주세요.')

y = ft_y.reshape((-1))
x = np.array(list(zip(mfcc_x, ft_x)))
xtrain, xtest, ytrain, ytest = pre.preprocess(xydata=[x, y])  # 전처리

mfcc_xtrain, ft_xtrain = xtrain[:,0], xtrain[:,1]
mfcc_xtest, ft_xtest = xtest[:,0], xtest[:,1]
mfcc_xtrain = mfcc_xtrain.reshape((-1, 30, 100))
mfcc_xtest = mfcc_xtest.reshape((-1, 30, 100))
ft_xtrain = ft_xtrain.reshape((-1, 30, 100))
ft_xtest = ft_xtest.reshape((-1, 30, 100))

# 데이터 불균형 문제 해결을 위해 class_weight 사용
class_weight = [(1 / pre.get_count(ytrain, [1, 0]))*len(ytrain)/2.0,
                (1 / pre.get_count(ytrain, [0, 1]))*len(ytrain)/2.0]

# 모델 정의
model = models.ClassificationModel().build_model()
try:
    # 이어서 학습
    model.load_weights('models/weights.h5')
except:
    pass

# 학습
model.fit([mfcc_xtrain, ft_xtrain], ytrain, validation_data=([mfcc_xtest, ft_xtest], ytest),
          batch_size=28, epochs=100, verbose=1, shuffle=True, class_weight=class_weight)

# 모델 저장
model.save_weights('models/weights.h5')
