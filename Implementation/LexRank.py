from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import pandas as pd
import csv
import nltk
# nltk.download('punkt')

data=pd.read_csv("../Dataset/ReleaseNoteDataset.csv")
index = 0
notesList = data['notes']
commits = data["commitMsg"]

def releaseSList(string):
    li = list(string.split("\n"))
    return li

def datawrite(summary, i):
    myFile = open('dataLx.csv', 'a', newline='')
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
    summarizer_lx = LexRankSummarizer()
    summary = ""
    topsentence = rnLength
    # Extract top 10 sentences as the summary
    print(topsentence)
    # if rnLength > len(ranked_sentences):
    #     topsentence = len(ranked_sentences)
    # else:
    #     topsentence = rnLength
    summary_lx = summarizer_lx(parser.document, topsentence)
    print(type(summary_lx))



    for sen in summary_lx:
        summary = summary + str(sen) + ". "
    print(summary)
    datawrite(summary, index)
    index = index + 1

