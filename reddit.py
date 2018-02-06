import json
import numpy
import sqlite3 as sqlite
import datetime
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

##Extract comments from the database, and integrate the comments based on the link id 
conn = sqlite.connect('reddit-opiates.db')
database=conn.cursor()
comment_group={}
database.execute('SELECT COUNT(*), link_id, group_concat(body) FROM Comment GROUP BY Comment.link_id')
for j in database:
	comment_group[j[1]]={'count':j[0],'content':j[2]}

# print len(comment_group.keys())

#Save the comments group into a document list
doc_complete=[]
for doc in comment_group.keys():
	doc_complete.append(comment_group[doc]['content'])
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=1)

for i in  ldamodel.show_topics(num_topics=5, num_words=25, log=False, formatted=True):
    print i[0], i[1]
# print(ldamodel.print_topics(num_topics=5, num_words=20))