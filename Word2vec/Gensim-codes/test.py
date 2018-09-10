import codecs

f=codecs.open('../wiki_data/zhwiki_ch_big_after_jiebaCut.txt','r',encoding="utf8")
line=f.readline()
print(line)