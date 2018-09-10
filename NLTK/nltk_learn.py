import nltk
import jieba
import re
from nltk import word_tokenize
from nltk.corpus import sinica_treebank

sinica_text = nltk.Text(sinica_treebank.words())
print(sinica_text.concordance('我'))

## 推特特殊语言体定义
emoticons_str = r"""
(?:
[:=;] # 眼睛
[oO\-]? # 鼻鼻子子
[D\)\]\(\]/\\OpP] # 嘴
)"""
regex_str = [
emoticons_str,
r'<[^>]+>', # HTML tags
r'(?:@[\w_]+)', # @某人人
r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # 话题标签
r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
# URLs
r'(?:(?:\d+,?)+(?:\.?\d+)?)', # 数字
r"(?:[a-z][a-z'\-_]+[a-z])", # 含有 - 和 ‘ 的单词
r'(?:[\w_]+)', # 其他
r'(?:\S)' # 其他
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
def tokenize(s):
    return tokens_re.findall(s)
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

tweet = 'RT @angelababy: love you baby! :D http://ah.love #168cm'
print(word_tokenize(tweet))
print(preprocess(tweet))
# ['RT', '@angelababy', ':', 'love', 'you', 'baby',
# ’!', ':D', 'http://ah.love', '#168cm']

