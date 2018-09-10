import nltk
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

sentense = 'hello , nice to meet you, how are you'
words = nltk.word_tokenize(sentense)
stopwords = stopwords.words('english')
filtered_words = [word for word in words if word not in stopwords]
# print(filtered_words)
print(stopwords)