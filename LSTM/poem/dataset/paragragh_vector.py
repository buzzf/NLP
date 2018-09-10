#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: zzf
# @Date:   2018-09-01 19:42:41
# @Last Modified by:   zzf
# @Last Modified time: 2018-09-01 23:35:39

from gensim.models import Word2Vec
import numpy as np 

# load word2vec model
# w2v = Word2Vec.load('poems.model')

def word_2_vec(w2v):
	word2vec_dict = {}
	words = list(w2v.wv.vocab.keys())
	for word in words:
		word2vec_dict[word] = w2v.wv[word]
	return words, word2vec_dict


# load raw data
def read_data(cutfile):
	raw_data = []
	with open(cutfile, 'r') as f:
		for line in f.readlines():
			sentense = line.strip().split(' ')
			raw_data.append(sentense)
	return raw_data



# change to sentense vector

def sent2vec(sen, w2v):
	wv = []
	for w in sen:
		try:
			wv.append(w2v.wv[w])
		except:
			continue
	wv = np.array(wv)
	sv = wv.mean(axis=0)
	return sv

def poem_vector(w2v, cutfile):
	raw_data = read_data(cutfile)
	poemvector = []
	vec2sen_dict = {}
	sen2vec_dict = {}

	for word in raw_data:
		try:
			newline = sent2vec(word, w2v).tolist()
			poemvector.append(newline)
			vec2sen_dict[tuple(newline)] = ''.join(word)
			sen2vec_dict[''.join(word)] = newline
		except:
			continue
	return poemvector, vec2sen_dict, sen2vec_dict

def voc_vec_return(path, cutfile):
	w2v = Word2Vec.load(path)
	words, word2vec_dict = word_2_vec(w2v)
	poemvector, vec2sen_dict, sen2vec_dict = poem_vector(w2v, cutfile)
	return poemvector, word2vec_dict, words

