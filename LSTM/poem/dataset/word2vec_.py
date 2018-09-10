#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: zzf
# @Date:   2018-08-31 19:31:57
# @Last Modified by:   zzf
# @Last Modified time: 2018-08-31 23:40:24

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing



model = Word2Vec(LineSentence('./data/poems_cut.txt'), size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())
model.save('poems.model')
model.wv.save_word2vec_format('poems.vector', binary=False)
