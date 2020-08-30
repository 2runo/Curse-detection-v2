# MFCC embedding을 수행한다.
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Bidirectional, Dropout, GRU, Reshape, BatchNormalization, \
    LeakyReLU, Activation
import hgtk
from char2vec import CHAR2VEC


class MfccEmbedding():
    def __init__(self, enc=None, weights_path="embedding_models/mfcc.h5"):
        self.enc = enc
        self.weights_path = weights_path

    def embedding(self, words):
        # mfcc 모델 예측 결과를 반환한다.
        if not self.enc:
            self.enc = self.get_encoder()
        vec = np.array([self.vectorize(self.decompose(w)) for w in words])  # 임베딩
        x = self.padding(vec, 80)  # 패딩
        return self.enc.predict(x)  # 예측

    def distance(self, a, b):
        # a, b의 유클리드 거리를 구한다.
        return np.linalg.norm(a - b)

    def most_similar(self, a, words, topn=10):
        # a와 가장 발음 비슷한 words 순위
        embed = self.embedding([a] + words)
        return sorted(words, key=lambda x: self.distance(embed[0], embed[words.index(x)+1]))[:topn]

    def get_model(self):
        # 모델을 반환한다.
        inp = Input(shape=(80, 8))
        inter = Dense(128)(inp)
        inter = BatchNormalization()(inter)
        inter = Activation(LeakyReLU())(inter)
        inter = Dropout(0.3)(inter)
        inter = Bidirectional(GRU(32, dropout=0.4), merge_mode='concat')(inter)
        inter = Dense(100, activation='tanh')(inter)
        inter = Dropout(0.4)(inter)
        inter = Dense(400)(inter)

        model = Model(inp, inter)
        model.load_weights(self.weights_path)
        return model

    def get_encoder(self, model=None):
        # 인코더 모델을 반환한다.
        if self.enc:
            return self.enc
        if not model:
            model = self.get_model()

        self.enc = Model(model.input, model.layers[6].output)
        return self.enc

    def decompose(self, text, empty_char='-'):
        # 한글을 자모로 분리한다.
        result = []

        for c in list(text):
            if hgtk.checker.is_hangul(c):

                if hgtk.checker.is_jamo(c):
                    result.append(c + 2 * empty_char)
                else:
                    jamo = "".join(hgtk.letter.decompose(c))
                    result.append(jamo + (3 - len(jamo)) * empty_char)

            elif c == ' ':
                result.append(3 * empty_char)
            else:
                result.append(c + 2 * empty_char)

        return ''.join(result)

    def vectorize(self, letters):
        # 임베딩한다.
        letters = list(letters)
        result = []
        for l in letters:
            try:
                result.append(CHAR2VEC[l.lower()])
            except KeyError:
                result.append(CHAR2VEC['~'])
        return result

    def padding(self, x, length=128):
        # 패딩을 수행
        return tf.keras.preprocessing.sequence.pad_sequences(x, dtype='float32', maxlen=length, padding="post")


if __name__ == "__main__":
    emb = MfccEmbedding()
    while True:
        # 입력한 단어와 발음이 비슷한 단어 순으로 출력
        print(emb.most_similar(input(':'), ['이루노', '딥 러닝', '도서', '회사', '사건', '어린이', '청소년', '국가', '하루', '눈동자', '사실',
                                            '가족', '정부', '그룹', '손톱', '바나나', '문제', '질문', '숫자', '학교', '사과', '시간', 'study',
                                            'life', 'family', 'moon']))
