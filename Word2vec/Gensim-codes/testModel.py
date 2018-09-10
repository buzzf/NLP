from gensim.models import Word2Vec

model = Word2Vec.load('wiki.zh.big.model')

testwords = ['春天','主席','人工智能','手机','中国','美女','百度','佛教']
for i in range(len(testwords)):
    res = model.most_similar(testwords[i])
    print(testwords[i])
    print(res)
