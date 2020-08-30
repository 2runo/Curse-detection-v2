import extract_data as ext
import embedding as emb
import models
import numpy as np


class CurseDetector:
    def __init__(self, weights_paths):
        # 예측할 때는 아래 모델들을 사용하여 예측한다. (앙상블 기법)
        self.list_of_models = [self.get_model(path) for path in weights_paths]

        # 어텐션 모델
        self.attention_models_mfcc = []  # mfcc 단의 attention
        self.attention_models_ft = []  # fasttext 단의 attention
        for i, model in enumerate(self.list_of_models):
            self.attention_models_mfcc.append(self.get_attention_model(model, 0))
            self.attention_models_ft.append(self.get_attention_model(model, 1))

    def get_model(self, weights_path):
        model = models.ClassificationModel().build_model()
        model.load_weights(weights_path)
        return model

    def get_attention_model(self, model, n):
        # model -> attention weights을 반환하는 모델로 변환한다.
        outs = models.ClassificationModel().attention_block(attention_only=True)
        attention_model = models.Model(inputs=outs[0], outputs=outs[1])
        for i in range(len(attention_model.layers)):
            attention_model.layers[i].set_weights(model.layers[i*2+n].get_weights())
        return attention_model

    def embed(self, text, y=None, return_tags=False):
        text = ext.long2short(text)  # 연속적인 글자 단축

        if y is None:
            ft_x, tags = emb.embedding_fasttext(text, return_tags=return_tags)  # fasttext 임베딩
        else:
            (ft_x, y), tags = emb.embedding_fasttext(text, y, return_tags=return_tags)  # fasttext 임베딩
        mfcc_x = emb.embedding_mfcc_tags(tags)  # mfcc 임베딩
        ft_x = ft_x.reshape((-1, 30, 100))
        mfcc_x = mfcc_x.reshape((-1, 30, 100))

        if return_tags:
            return mfcc_x, ft_x, y, tags
        else:
            return mfcc_x, ft_x, y

    def attention_predict(self, i, x1, x2):
        # 평균 어텐션 스코어를 반환한다.
        return (self.attention_models_mfcc[i].predict(x1) + self.attention_models_ft[i].predict(x2)) / 2

    def predict(self, text):
        text = ext.long2short([text])[0]  # 연속적인 글자 단축

        ft_x, tags = emb.embedding_fasttext([text], return_tags=True)  # fasttext 임베딩
        if tags == []:
            return [[None]]
        mfcc_x = emb.embedding_mfcc_tags(tags)  # mfcc 임베딩
        ft_x = ft_x[0].reshape((1, 30, 100))
        mfcc_x = mfcc_x[0].reshape((1, 30, 100))
        tags = tags[0]

        if ft_x is None or mfcc_x is None:
            return None

        # [(모델1 예측 결과, 어텐션 스코어, 단어 목록), (모델2 ..), ..] 반환
        return [(model.predict([mfcc_x, ft_x])[:,1].reshape(-1),
                 self.attention_predict(i, mfcc_x, ft_x).reshape(-1)[:len(tags)],
                 tags) for i, model in enumerate(self.list_of_models)]

    def replace_ignore_space(self, text, to, replace):
        # 띄어쓰기를 무시하고 replace한다.
        # ex) f('안 녕하세요', '안녕', '*') -> '*하세요'
        i = 0
        ing = False
        start_i = 0
        ing_i = 0
        for c in text:
            if ing_i == len(to):
                return text.replace(text[start_i:i], replace)
            if c == ' ':
                i += 1
                continue
            if c == to[ing_i]:
                if not ing:
                    start_i = i
                    ing_i = 0
                    ing = True
                ing_i += 1
            else:
                ing_i = 0
                ing = False
                start_i = 0
                if c == to[ing_i]:
                    if not ing:
                        start_i = i
                        ing_i = 0
                        ing = True
                    ing_i += 1
            i += 1
        if ing_i == len(to):
            return text.replace(text[start_i:i], replace)

    def masking(self, text):
        # 텍스트를 마스킹하여 (*처리) 반환한다.
        while True:
            pred = self.ensemble(text, return_attention=True)

            if pred[0] <= 0.5:
                return text
            text = self.replace_ignore_space(text, pred[2][np.argmax(pred[1])], '*')

    def evaluate(self, path, mode='each'):
        # path의 라벨링 돼 있는 텍스트 데이터 파일을 읽어서 모델을 평가한다.
        x, y = ext.load_data(path)
        y = [int(i) for i in y]

        mfcc_x, ft_x, y, tags = self.embed(x, y, return_tags=True)

        preds = [model.predict([mfcc_x, ft_x])[:,1] for model in self.list_of_models]  # 예측
        if mode == 'each':
            # 각각의 모델 output에 대하여 평가하기
            accs = []
            for model_n, pred in enumerate(preds):
                acc = models.np_acc(y.reshape(-1), pred.reshape(-1))
                accs.append(acc)
        elif mode == 'ensemble':
            # 모든 모델 output의 평균으로 하나의 값을 통합하여 평가하기
            pred = np.average(preds, axis=0)
            accs = models.np_acc(y, pred.reshape(-1))

        return accs

    def ensemble(self, text, return_attention=False):
        # 모든 예측 결과의 평균 반환
        pred = self.predict(text)
        if return_attention:
            # 어텐션 스코어 반환?
            return [np.average([i[c] for i in pred], axis=0) for c in range(2)] + [pred[-1][2]]
        else:
            return np.average([i[0] for i in pred], axis=0)[0]


if __name__ == "__main__":
    # 예측할 때는 weights_paths의 모델들을 사용하여 예측한다. (앙상블 기법)
    weights_paths = ['models/weights.h5', 'models/weights2.h5', 'models/weights3.h5']
    curse = CurseDetector(weights_paths)

    print(curse.masking('씌바 벌레같은 놈아 안죽냐?'))       # '* *같은 *아 안죽냐?'
    print(curse.masking('옷 다릴 때 니 뇌도 같이 다렸니?'))  # '* * 때 니 *도 같이 다렸니?'
    print(curse.ensemble('니입에서짐승소리가들린다'))        # 0.78354186
    while True:
        text = input(':')

        print(curse.ensemble(text))
        print(curse.masking(text))
