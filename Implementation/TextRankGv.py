import csv
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv("../Dataset/ReleaseNoteDataset.csv")
index = 0


def commitSList(string):
    li = list(string.split("\\n"))
    return li

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def releaseSList(string):
    li = list(string.split("\n"))
    return li

def datawrite(summary, i):
    myFile = open('dataTx.csv', 'a', newline='')
    with myFile:
        fieldnames = ['summary']
        writer = csv.DictWriter(myFile, fieldnames=fieldnames)
        if(i==0):
            writer.writeheader()
        writer.writerow({'summary':summary})
    return

commitList = data['commitMsg']
notesList = data['notes']

for x in range(len(notesList)):
    note = notesList[x]
    commit = commitList[x]
    rn = releaseSList(note)
    sentences = []
    sentences = commitSList(commit)
    rnLength = len(rn)
    word_embeddings = {}
    f = open('../glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = stopwords.words('english')
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = ""
    topsentence = 1
    if rnLength > len(ranked_sentences):
        topsentence = len(ranked_sentences)
    else:
        topsentence = rnLength
    # Extract top 10 sentences as the summary
    print(topsentence)
    for sen in range(topsentence):
        summary = summary + ranked_sentences[sen][1] + ". "
    datawrite(summary, index)
    index = index + 1