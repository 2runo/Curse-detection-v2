# 댓글 데이터를 전처리한다.
# 긴 댓글은 제거하거나 특수문자를 제거한다.
import re
import itertools


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


def long2short(x):
    # 연속적으로 긴 단어는 간추리기
    # ef) f('ㅋㅋㅋㅋㅋㅋㅋ앜ㅋㅋㅋ') -> f('ㅋㅋ앜ㅋㅋ')
    result = []
    for ele in x:
        while True:
            candidates = set(re.findall(r'(\w)\1', ele))
            repeats = itertools.chain(*[re.findall(r"({0}{0}+)".format(c), ele) for c in candidates])

            keep = False
            for org in [i for i in repeats if len(i) >= 3]:
                ele = ele.replace(org, org[0]*2)
                keep = True
            if not keep:
                break
        result.append(ele)
    return result


def cut_long(x, y, maxlen=128):
    # maxlen보다 길이가 긴 데이터는 없앤다. + 빈 데이터도 없앤다. ('' -> 제거)
    resultx = []
    resulty = []
    for n in range(len(x)):
        if len(x[n]) == 0:
            # 빈 데이터 제거
            continue
        if len(x[n]) <= maxlen:
            resultx.append(x[n])
            resulty.append(y[n])
    return resultx, resulty


def dedup(x, y):
    # x의 중복을 제거한다.
    resultx = []
    resulty = []
    for n in range(len(x)):
        if not x[n] in resultx:
            resultx.append(x[n])
            resulty.append(y[n])
    return resultx, resulty


def preprocessing_data(path='dataset/data.txt'):
    x, y = load_data(path)  # 데이터 로드
    x = long2short(x)  # 연속적인 글자 단축 (ㅋㅋㅋㅋ->ㅋㅋ)
    x, y = dedup(x, y)  # x 중복 제거
    x, y = cut_long(x, y, maxlen=1500)  # 길이 긴 데이터, 빈 데이터 제거
    return x, y
