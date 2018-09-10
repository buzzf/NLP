# !/usr/bin/python
# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
from gensim.models import TfidfModel
from pprint import pprint

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    f = open('LDA_test.txt')
    stop_list = set('for a of the and to in'.split())
    # texts = [line.strip().split() for line in f]
    # print('Before')
    # pprint(texts)
    print('After')
    texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in f]
    print('Text = ')
    pprint(texts)

#   corpora是gensim中的一个基本概念，是文档集的表现形式，也是后续进一步处理的基础
#   corpora.Dictionary 得到一个词典

    dictionary = corpora.Dictionary(texts)  
    print('dictionary: ', dictionary)
    V = len(dictionary)

#   doc2bow(text) 将text文本转换稀疏矩阵, [(0,1), (1,1)],表明id为0,1的词汇出现了1次    

    corpus = [dictionary.doc2bow(text) for text in texts]
    print('bow: ', corpus)

#   models.TfidfModel 转换成词频-逆文本词频值  格式为[(0, 0.4301019571350565), (1, 0.4301019571350565)]

    corpus_tfidf = TfidfModel(corpus)[corpus]
    print('TF-IDF:')
    for c in corpus_tfidf:
        print(c)


#### 以下也可以直接将corpus喂给LDA

#   LSI主题模型

    print('\nLSI Model:')
    lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word=dictionary) # 得到主题模型
    topic_result = [a for a in lsi[corpus_tfidf]]   # 将文档放入模型，得到所有文档的主题
    pprint(topic_result)
    print('LSI Topics:') # 打印2个主题的前5个词
    pprint(lsi.print_topics(num_topics=2, num_words=5))
    similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])   # similarities.Similarity()
    similarity2 = similarities.MatrixSimilarity(corpus_tfidf)   # 不考虑主题模型

    print('Similarity:')
    pprint(list(similarity))
    pprint(list(similarity2))

#   LDA主题模型

    print('\nLDA Model:')
    num_topics = 2

#   minimum_probability=0.001 阈值，如果prob小于0.001则排除这一项，为了控制向量长短的

    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001, passes=10)
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]  # 将文档放入模型，得到所有文档的主题
    print('Document-Topic:\n')
    pprint(doc_topic)

    print('another way to get Document-Topic:')
    for doc_topic in lda.get_document_topics(corpus_tfidf):    # 所有文档的主题分布
        print(doc_topic)

    for topic_id in range(num_topics):
        print('Topic', topic_id)
        # pprint(lda.get_topic_terms(topicid=topic_id))
        pprint(lda.show_topic(topic_id))

    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print('Similarity:')
    pprint(list(similarity))

#   结构化的hdp模型
    hda = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hda[corpus_tfidf]]
    print('\n\nUSE WITH CARE--\nHDA Model:')
    pprint(topic_result)
    print('HDA Topics:')
    print(hda.print_topics(num_topics=2, num_words=5))
