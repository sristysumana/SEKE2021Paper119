from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import pandas as pd
import csv
import nltk
nltk.download('punkt')

data=pd.read_csv("../Dataset/ReleaseNoteDataset.csv")
index = 0
notesList = data['notes']
commits = data["commitMsg"]

def releaseSList(string):
    li = list(string.split("\n"))
    return li

def datawrite(summary, i):
    myFile = open('../Code/dataLSA.csv', 'a', newline='')
    with myFile:
        fieldnames = ['summary']
        writer = csv.DictWriter(myFile, fieldnames=fieldnames)
        if(i==0):
            writer.writeheader()
        writer.writerow({'summary':summary})
    return

for x in range(len(notesList)):
    rn = releaseSList(notesList[x])
    rnLength = len(rn)
    parser = PlaintextParser.from_string(commits[x],Tokenizer("english"))
    summarizer_lsa = LsaSummarizer()
    summary = ""
    topsentence = rnLength
    # Extract top 10 sentences as the summary
    print(topsentence)
    # if rnLength > len(ranked_sentences):
    #     topsentence = len(ranked_sentences)
    # else:
    #     topsentence = rnLength
    summary_lsa = summarizer_lsa(parser.document, topsentence)
    print(type(summary_lsa))



    for sen in summary_lsa:
        summary = summary + str(sen) + ". "
    print(summary)
    datawrite(summary, index)
    index = index + 1

