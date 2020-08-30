# 댓글을 임베딩한다.
import tensorflow as tf
import numpy as np
import joblib
from konlpy.tag import Okt
import fasttext
from mfcc_encoder import MfccEmbedding
import hgtk
import extract_data


# fasttext
ft = fasttext.load_model('embedding_models/fasttext.bin')
ft_dimension = ft.get_dimension()

# mfcc
mfcc = MfccEmbedding()
mfcc.get_encoder()

# 형태소 분석기
okt = Okt()


def decompose(text, empty_char='-'):
    # 한글을 자모로 분리한다.
    # ex) f('아녕') -> 'ㅇㅏ-ㄴㅕㅇ'
    result = []
    for c in list(text):
        if hgtk.checker.is_hangul(c):
            if hgtk.checker.is_jamo(c):
                result.append(c + 2*empty_char)
            else:
                jamo = "".join(hgtk.letter.decompose(c))
                result.append(jamo + (3-len(jamo))*empty_char)
        elif c == ' ':
            result.append(3*empty_char)
        elif c == '|' or c == '\n':
            # '|'이면 그대로 유지 (품사 경계 나누는 선임)
            result.append(c.replace('|', ' '))
        else:
            result.append(c + 2*empty_char)
    return ''.join(result)


def load_data(path='dataset/data.txt'):
    # 데이터를 로드한다.
    with open(path, 'r', encoding='utf8') as f:
        raw = f.read().replace('\n\n', '\n')

    lines = [line.split('|') for line in raw.split('\n')]

    x, y = [], []
    for line in lines:
        if len(line) <= 1:
            print('비정상적인 데이터가 %s에서 감지되었습니다. 기대한 형태: "안녕하세요|0"\t실제 형태: "%s"' % (path, '|'.join(line)))
        if len(line) >= 3:
            line = ['|'.join(line[:-1]), line[-1]]
        x.append(line[0])
        y.append(line[1])

    return x, y


def split_tag(text):
    # 문장(text)을 단어로 나눠서 반환한다.
    # ex) f('안녕 나는 조준희야') -> ['안녕', '나는', '조준희야']
    return [i[0] for i in okt.pos(text)]


def slice_as_lengths(lst, lengths):
    # 리스트를 길이들로 쪼갠다.
    # ex) f([1,2,3,4,5,6,7,8,9], [2,4,3]) -> [[1,2], [3,4,5,6], [7,8,9]]
    r = []
    for i, l in enumerate(lengths):
        r.append(lst[sum(lengths[:i]):sum(lengths[:i+1])])
    return r


def embedding_fasttext(x, y=None, length=3000, return_tags=False):
    # fasttext 임베딩을 수행한다.
    # 레이블 데이터라면 y도 같이 넣어 줘야 순서가 안 섞임 (긴 문장은 자동으로 탈락시키기 때문)
    # return_tags : True면 단어 별로 나뉜 리스트도 같이 반환 (예측에 사용)
    result = []
    result_y = []
    if return_tags:
        result_tags = []

    for n, text in enumerate(x):
        tmp = np.array([])
        splited = split_tag(text)
        if len(splited) * ft_dimension > length:
            continue
        for word in splited:
            try:
                tmp = np.concatenate((tmp, ft.get_word_vector(decompose(word))))
            except:
                pass
        if return_tags:
            result_tags.append(splited)
        result.append(np.concatenate((tmp, [0.] * (length - len(tmp)))))
        if y:
            result_y.append(y[n])

    result = np.array(result)
    result_y = np.array(result_y)
    if y:  # 반환 값은 y == None이면 result, 아니면 (result, y)
        result = (result, result_y)
    if return_tags:
        return result, result_tags
    return result


def embedding_mfcc(x, y=None, length=3000, return_tags=False):
    # mfcc 임베딩을 수행한다.
    # 레이블 데이터라면 y도 같이 넣어줘야 순서가 안 섞임 (긴 문장은 자동으로 탈락시키기 때문)
    # return_tags : True면 단어 별로 나뉜 리스트도 같이 반환 (예측에 사용)
    result_y = []
    if return_tags:
        result_tags = []

    will = []
    will_lengths = []
    for n, text in enumerate(x):
        splited = split_tag(text)
        if len(splited) * ft_dimension > length:
            continue
        if return_tags:
            result_tags.append(splited)
        if y:
            result_y.append(y[n])

        will += splited
        will_lengths.append(len(splited))

    result = mfcc.embedding(will).reshape((-1, 100))
    result = np.array(slice_as_lengths(result, will_lengths))
    result = [i.reshape(-1) for i in result]
    result = padding(result, length)

    result_y = np.array(result_y)
    if y:
        result = (result, result_y)
    if return_tags:
        return result, result_tags
    return result  # 반환 값은 y 받았으면 (result, y)  y 안 받았으면 result


def embedding_mfcc_tags(tags, length=3000):
    # tags가 있는 embedding_mfcc()
    will = []
    will_lengths = []
    for n, splited in enumerate(tags):
        if len(splited) * ft_dimension > length:
            continue

        will += splited
        will_lengths.append(len(splited))

    result = mfcc.embedding(will).reshape((-1, 100))
    result = np.array(slice_as_lengths(result, will_lengths))
    result = [i.reshape(-1) for i in result]
    result = padding(result, length)

    return result


def div_length(length):
    # length를 둘로 나눔 (단, 두 값의 합은 length여야 함)
    # ex) f(11) -> 5, 6
    a, b = length // 2, length // 2
    if a + b != length:
        b += 1
    return a, b


def padding(x, length=128):
    # 패딩을 수행
    return tf.keras.preprocessing.sequence.pad_sequences(x, dtype='float32', maxlen=length, padding="post")


if __name__ == "__main__":
    x, y = extract_data.preprocessing_data('dataset/data.txt')  # 데이터 전처리 수행
    y = list(map(int, y))
    mfcc_x, mfcc_y = embedding_mfcc(x, y)
    print(mfcc_x.shape, mfcc_y.shape)
    ft_x, ft_y = embedding_fasttext(x, y)
    print(ft_x.shape, ft_y.shape)
    if len(mfcc_x) != len(mfcc_y):
        raise Exception('mfcc_x와 mfcc_y의 길이가 다릅니다. 어딘가에 문제가 있는 것 같습니다.')
    if len(ft_x) != len(ft_y):
        raise Exception('ft_x와 ft_y의 길이가 다릅니다. 어딘가에 문제가 있는 것 같습니다.')

    joblib.dump([mfcc_x, mfcc_y, ft_x, ft_y], 'dataset/data.joblib')
