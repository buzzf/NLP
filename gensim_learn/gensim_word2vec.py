from gensim.models import word2vec

"""
利用gensim.models.Word2Vec(sentences)建立词向量模型

该构造函数执行了三个步骤：
建立一个空的模型对象，遍历一次语料库建立词典，第二次遍历语料库建立神经网络模型
可以通过分别执行 model=gensim.models.Word2Vec()，model.build_vocab(sentences)，model.train(sentences)来实现

训练时可以指定以下参数:
min_count: 指定了需要训练词语的最小出现次数，默认为5
size:      指定了训练时词向量维度，默认为100
worker:    指定了完成训练过程的线程数，默认为1不使用多线程。只有注意安装Cython的前提下该参数设置才有意义

可以通过model.save('fname')或model.save_word2vec_format(fname)来保存模型为文件，
使用model.load(fname)或model.load_word2vec_format(fname,encoding='utf-8')来加载模型，供查询


"""

raw_sentences = ['The reason for separating the trained vectors into KeyedVectors is that if you don’t need the full model state any more (don’t need to continue training), the state can discarded, resulting in a much smaller and faster object that can be mmapped for lightning fast loading and sharing the vectors in RAM between processes']

sentences = [s.split() for s in raw_sentences]
print(sentences)

# train
model = word2vec.Word2Vec(sentences, size=30, min_count=1)

###########################
#     model save          #
###########################

#  该方法保存的文件不能利用文本编辑器查看但是保存了训练的全部信息，可以在读取后追加训练

# model.save('MyModel')

# 该方法保存为word2vec文本格式但是保存时丢失了词汇树等部分信息，不能追加训练

# model.wv.save_word2vec_format('mymodel.txt', binary=False)
# model.wv.save_word2vec_format('mymodel.bin.gz', binary=True)

# 追加训练
model2 = word2vec.Word2Vec.load('MyModel')
# model2.train(more sentences)


###########################
#     model 使用          #
###########################

# 获取词向量
print(model['that'])

# 计算一个词最相近的词，倒排序
print(model.most_similar(['vectors']))
print(model.most_similar(positive=['reason','vectors'], negative=['object']))

# 计算两个词之间的余弦相似度
simil = model.similarity('reason', 'vectors')
print(simil)

# 计算两个集合之间的余弦相似度
list1 = ['the', 'reason', 'for', 'separating'] 
list2 = ['the', 'reason', 'to', 'separating'] 
list_sim1 = model.n_similarity(list1,list2)
print(list_sim1)


# 选出集合中不同类的词语
model.doesnt_match('vectors KeyedVectors you'.split())
