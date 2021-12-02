import re

from utils import logger

BIO_PATTERN = re.compile(r'\[(.+?)\](_[A-Z]{3}_)')
BIO_BEGIN = re.compile(r'_[A-Z]{3}_\[')
BIO_END = re.compile(r'\]_[A-Z]{3}_')


def pre_process(text):
    text = BIO_PATTERN.sub(r'\2[\1]\2', text)
    i = 0
    res = []
    label = ''
    is_begin = False
    is_inner = False
    length = len(text)
    while i < length:
        c = text[i]
        if '_' == c and i + 5 < length and BIO_BEGIN.fullmatch(text, i, i + 6):
            label = text[i + 1:i + 4]
            i += 6
            is_begin = True
            is_inner = True
            continue
        if ']' == c and i + 5 < length and BIO_END.fullmatch(text, i, i + 6) and is_inner:
            i += 6
            is_inner = False
            continue

        if is_begin:
            res.append([c, f'B-{label}'])
            is_begin = False
        elif is_inner:
            res.append([c, f'I-{label}'])
        else:
            res.append([c, 'O'])
        i += 1
    return res


def post_process(text, predict):
    res = []
    end_idx = len(text) - 1
    for i in range(end_idx):
        token = text[i]
        label1 = predict[i]
        label2 = predict[i + 1]
        if 'O' == label1:
            res.append(token)
        else:
            if label1.startswith('B-'):
                res.append('[')
            res.append(token)
            if not label2.startswith('I-'):
                res.append(f']_{label1[2:5]}_')
    token = text[end_idx]
    label1 = predict[end_idx]
    if label1.startswith('B-'):
        res.append('[')
    res.append(token)
    if 'O' != label1:
        res.append(f']_{label1[2:5]}_')
    return ''.join(res)


if __name__ == '__main__':
    text = '我喷了[chanel]_PRO_香水，在[澳大利亚]_LOC_吃着[肯德基]_COM_炸鸡'
    bio = pre_process(text)
    labeled = post_process([char[0] for char in bio], [char[1] for char in bio])
    logger.info(text)
    logger.info(bio)
    logger.info(labeled)
    logger.info(f'check status: {text == labeled}')
