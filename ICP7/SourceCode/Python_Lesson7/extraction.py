import requests
from bs4 import BeautifulSoup
import nltk
import re


wiki = "https://en.wikipedia.org/wiki/Google"
page = requests.get(wiki).text
soup = BeautifulSoup(page,"html.parser")
#print(soup.prettify())
text = str(soup.encode("UTF-8"))
file = open("input.txt", "w")
file.write(text)
file.close()
nltk.download('punkt')

print('-----------------------Tokenization-------------------------------------')
'''for s in sTokens:'''
for k in text.split("\n"):
    text1 = str(re.sub(r"[^a-zA-Z0-9]+", ' ', k))
    file = open("input1.txt", "w")
    file.write(text1)

sTokens = nltk.word_tokenize(text1)
for s in sTokens:
    print(s)

nltk.download('averaged_perceptron_tagger')
print('----------------------Parts of speech----------------------------')
print(nltk.pos_tag(sTokens));

print('------------------------------Lemmitization--------------------------')
from nltk.stem import  WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize(text1))

print('------------------------------NameEntity--------------------------')
from  nltk import wordpunct_tokenize ,pos_tag,ne_chunk
nltk.download('wordnet')
nltk.download('words')
nltk.download('maxent_ne_chunker')
print(ne_chunk(pos_tag(wordpunct_tokenize(text1))))

print('------------------------------Stemming--------------------------')
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

pStemmer = PorterStemmer()
lStemmer = LancasterStemmer()

n1 = 0
for t in sTokens:
    n1 = n1 + 1
    if n1 < 4:
        print(pStemmer.stem(t), lStemmer.stem(t))
print('-------------------trigrams-------------------------------')
from nltk.util import ngrams
token = nltk.word_tokenize(text1)
for s in sTokens:
        token = nltk.word_tokenize(s)
        bigrams = list(ngrams(token, 2))
        trigrams = list(ngrams(token, 3))
        print("The text:", s, "\nword_tokenize:", token, "\nbigrams:", bigrams, "\ntrigrams", trigrams)






