#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 18.06.24
@author: leonhard.hennig@dfki.de
"""
import os

import nltk
import pandas as pd
from tqdm import tqdm

import uuid
import spacy
import argparse


#BASE_PATH="/ds/text/NarraSum"
#OUTPUT_PATH="/netscratch/hennig/code/eventsum/NarraSum_preprocessed"

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_path", default=None, type=str, required=True,
                        help="The input data dir. Should contain the train, test and validation JSON files for NarraSum.")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="The output data directory.")
    parser.add_argument("--field", default=None, type=str, required=True, help="Data field to be processed, either 'document' or 'summary'")
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    else:
        print('Warn: Overwriting existing output directory!')

    # spacy for tokenization
    #spacy.require_gpu()
    try:
       nlp = spacy.load("en_core_web_sm") 
       #, exclude=["parser", "attribute_ruler", "lemmatizer", "ner"]) #disable=["parser, senter, attribute_ruler, lemmatizer, ner"])
    except OSError:
       from spacy.cli.download import download as spacy_download
       spacy_download("en_core_web_sm")
       nlp = spacy.load("en_core_web_sm") 
       #exclude=["parser", "attribute_ruler", "lemmatizer", "ner"]) # disable=["parser, senter, attribute_ruler, lemmatizer, ner"])
    # nltk punkt for sentence segmentation: https://github.com/segment-any-text/wtpsplit sat is the new state of the art
    nltk.download("punkt")

    field = args.field
    for split in ["test", "train", "validation"]:
        print(f'Preprocessing {split}')
        fname = f"{args.input_path}/{split}.json"
        data = pd.read_json(path_or_buf=fname, lines=True)

        for i, doc in tqdm(enumerate(data[field]), total=len(data[field])):
            data.loc[i, field] = nltk.tokenize.sent_tokenize(doc)
#           data.document[i] = nltk.tokenize.sent_tokenize(data.summary[i])

        tokensList = []
#        sumTokensList = []
        for i, doc in tqdm(enumerate(data[field]), total=len(data[field])):
            sents = []
            for sent in doc:
                sents.append(nltk.word_tokenize(sent))
            tokensList.append(sents)
#            sum_sents = []
#            for sum_sent in data.summary[i]:
#                sum_sents.append(nltk.word_tokenize(sum_sent))
#            sumTokensList.append(sum_sents)
        data['tokens'] = tokensList
#        data['tokensSum'] = sumTokensList

        content = []
#        contentSum = []
        for ind in tqdm(data.index, total=len(data.index)):
            ls = []
            for sent, tok in zip(data[field][ind], data['tokens'][ind]):
                dic = {}
                dic['sentence'] = sent
                dic['tokens'] = tok
                ls.append(dic)
            content.append(ls)
#            sumLS = []
#            for sent, tok in zip(data['summary'][ind], data['tokensSum'][ind]):
#                dic = {}
#                dic['sentence'] = sent
#                dic['tokens'] = tok
#                sumLS.append(dic)
#            contentSum.append(sumLS)

        data['content'] = content
#        data['contentSum'] = contentSum

        data.drop("document", axis=1, inplace=True)
        data.drop("tokens", axis=1, inplace=True)
#        data.drop("tokensSum", axis=1, inplace=True)
        data.drop("summary", axis=1, inplace=True)

        # with open(f'{OUTPUT_PATH}/{split}.json', 'w') as f:
        #    f.write(data.to_json(orient='records', lines=True))

        # file_path1 = "/Users/clementgillet/Desktop/ready_for_pos/train.json"
        # trainNarraSum = pd.read_json(path_or_buf=file_path1, lines=True)

        # filter for PROPN, NOUN, VERB
        # Add entry as follows :
        # {"trigger_word": "Conquest", "sent_id": 0, "offset": [1, 2], "id": "f3d95fd23f790fb12875f8fe02bf5fb0"}

        candidatesList = []
        for q, elem in tqdm(enumerate(data.content), total=len(data.content)):
            candidates = []
            for i, sent in enumerate(elem):
                # print(sent['sentence'])
                doc = nlp(sent['sentence'])
                for j, w in enumerate(doc):
                    if w.pos_ in ["NOUN", "VERB", "ADJ"]:
                        dic = {}
                        dic["trigger-word"] = w.text
                        dic["sent_id"] = i
                        dic["offset"] = [j, j+1]
                        dic["id"] = str(uuid.uuid4()).replace("-","")
                        candidates.append(dic)
                        #print("(", w.text , ",", w.pos_, ")")
            #print(q/len(data)*100,"%")
            candidatesList.append(candidates)

        data["candidates"] = candidatesList


#        sumCandidatesList = []
#        for q, elem in tqdm(enumerate(data.contentSum), total=len(data.contentSum)):
#            candidatesSum = []
#            for i, sent in enumerate(elem):
#                # print(sent['sentence'])
#                doc = nlp(sent['sentence'])
#                for j, w in enumerate(doc):
#                    if w.pos_ in ["PROPN", "NOUN", "VERB", "AUX"]:
#                        dic = {}
#                        dic["trigger-word"] = w.text
#                        dic["sent_id"] = i
#                        dic["offset"] = [j, j+1]
#                        dic["id"] = str(uuid.uuid4()).replace("-","")
#                        candidatesSum.append(dic)
#                        #print("(", w.text , ",", w.pos_, ")")
#            #print(q/len(data)*100,"%")
#            sumCandidatesList.append(candidatesSum)
#
#        data["candidatesSum"] = sumCandidatesList

        with open(f"{args.output_path}/{split}_{field}.jsonl", 'w') as f:
            f.write(data.to_json(orient='records', lines=True))
    print('Done')


if __name__ == "__main__":
    main()
