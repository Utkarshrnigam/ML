import numpy as np
import pandas as pd
import sys

train = pd.read_csv("C:/Users/ASUS/Desktop/p/ML/ML/Dataset/naive_bayes/NLP_movieRating/Train.csv")
test = pd.read_csv("C:/Users/ASUS/Desktop/p/ML/ML/Dataset/naive_bayes/NLP_movieRating/Test.csv")

train = train.values
test = test.values

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer("[a-zA-Z]+")
ps = PorterStemmer()
stopwords_list = set(stopwords.words('english'))

def clean_data(review):
	review = review.lower()
	review = review.replace("<br /><br />"," ")
	tokens = tokenizer.tokenize(review)
	new_tokens = [token for token in tokens if token not in stopwords_list]
	tokens = [ps.stem(token) for token in new_tokens]
	review = ' '.join(tokens)
	return review


m = train.shape[0]
for i in range(m):
	train[i][0] = clean_data(train[i][0])

train_x = train[:,0]

corpus = []
for i in range(m):
	corpus.append(train_x[i])


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

print("done first step")

tf_Idfv = TfidfVectorizer()
vd = tf_Idfv.fit_transform(corpus).toarray()
print("done step two")


outputMovie = sys.argv[1]
out = open(outputMovie,'w',encoding="utf8")

for ix in vd:
	print((ix),file=out)

close(out)