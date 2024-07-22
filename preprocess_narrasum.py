#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 18.06.24
@author: leonhard.hennig@dfki.de
"""
import json
import os
import nltk
import uuid
import spacy
import argparse
import logging
import pandas as pd

from tqdm import tqdm
from itertools import islice

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.WARNING, format=FORMAT, datefmt="[%X]"
)
logger = logging.getLogger(__name__)


def preprocess_narrasum_nltk_spacy(input_path, output_path, field, spacy_model="en_core_web_sm"):
    # Clément's version

    # spacy mainly for PoS tagging
    # spacy.require_gpu()
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        from spacy.cli.download import download as spacy_download
        spacy_download(spacy_model)
        nlp = spacy.load(spacy_model)

    nltk.download("punkt")

    for split in ["test", "train", "validation"]:
        logger.info(f"Preprocessing {split}")
        fname = f"{input_path}/{split}.json"
        # Loads everything into memory, goes through every document 3 times
        data = pd.read_json(path_or_buf=fname, lines=True)

        # Segment sentences
        for i, doc in tqdm(enumerate(data[field]), total=len(data[field])):
            data.loc[i, field] = nltk.tokenize.sent_tokenize(doc)
        # Tokenize sentences
        tokens_list = []
        for i, doc in tqdm(enumerate(data[field]), total=len(data[field])):
            sents = []
            for sent in doc:
                sents.append(nltk.word_tokenize(sent))
            tokens_list.append(sents)
        data["tokens"] = tokens_list

        # Reorganize data
        content = []
        for ind in tqdm(data.index, total=len(data.index)):
            ls = []
            for sent, tok in zip(data[field][ind], data["tokens"][ind]):
                dic = {"sentence": sent, "tokens": tok}
                ls.append(dic)
            content.append(ls)

        data["content"] = content

        data.drop("document", axis=1, inplace=True)
        data.drop("tokens", axis=1, inplace=True)
        data.drop("summary", axis=1, inplace=True)

        # filter for PROPN, NOUN, VERB
        # Add entry as follows :
        # {"trigger_word": "Conquest", "sent_id": 0, "offset": [1, 2], "id": "f3d95fd23f790fb12875f8fe02bf5fb0"}

        # Process each sentence with spacy and extract event mention candidates (words belonging to NOUN, VERB or ADJ)
        candidates_list = []
        for q, elem in tqdm(enumerate(data.content), total=len(data.content)):
            candidates = []
            for i, sent in enumerate(elem):
                # logger.info(sent["sentence"])
                doc = nlp(sent["sentence"])
                for j, w in enumerate(doc):
                    if w.pos_ in ["NOUN", "VERB", "ADJ"]:
                        dic = {
                            "trigger-word": w.text,
                            "sent_id": i,
                            "offset": [j, j + 1],   # spacy token offsets will probably not match with nltk tokenization
                            "id": str(uuid.uuid4()).replace("-", "")
                        }
                        candidates.append(dic)
            candidates_list.append(candidates)

        data["candidates"] = candidates_list

        with open(f"{output_path}/{split}_{field}.jsonl", "w") as f:
            f.write(data.to_json(orient="records", lines=True))
    logger.info("Done")


def get_next_batch(fp, batch_size=200):
    for batch in iter(lambda: tuple(islice(fp, batch_size)), ()):
        docs = [json.loads(d) for d in batch]
        yield (
            [(d["document"], {"id": d["id"]}) for d in docs],
            [(d["summary"], {"id": d["id"]}) for d in docs]
        )


def preprocess_narrasum_spacy(input_path, output_path, spacy_model="en_core_web_trf"):
    # Leo's version: spaCy for everything
    try:
        nlp = spacy.load(spacy_model, disable=["attribute_ruler, lemmatizer, ner"])
    except OSError:
        from spacy.cli.download import download as spacy_download
        spacy_download(spacy_model)
        nlp = spacy.load(spacy_model, disable=["attribute_ruler, lemmatizer, ner"])

    for split in ["train", "test", "validation"]:
        logger.info(f'Preprocessing {split}')
        input_file = f"{input_path}/{split}.json"
        output_file_sum = f"{output_path}/{split}_summary.jsonl"
        output_file_doc = f"{output_path}/{split}_document.jsonl"
        with open(output_file_sum, 'w') as f_out_sum:
            with open(output_file_doc, 'w') as f_out_doc:
                with open(input_file, 'r') as f_in:
                    batch_size = 50
                    for doc_tuples in tqdm(get_next_batch(f_in, batch_size=batch_size)):
                        for field in ['document', 'summary']:
                            idx = 0 if field == 'document' else 1
                            spacy_doc_tuples = nlp.pipe(doc_tuples[idx], batch_size=batch_size, as_tuples=True)
                            for spacy_doc, context in spacy_doc_tuples:
                                candidates = []
                                for i, sent in enumerate(spacy_doc.sents):
                                    if len(sent) == 0 or len(sent.text.strip()) == 0:
                                        logger.warning(f'empty sentence')
                                        continue
                                    for j, tok in enumerate(sent):
                                        if tok.pos_ in ["NOUN", "VERB", "ADJ"]:
                                            dic = {
                                                "trigger-word": tok.text,
                                                "sent_id": i,
                                                "offset": [j, j + 1],
                                                "id": str(uuid.uuid4()).replace("-", "")
                                            }
                                            candidates.append(dic)
                                new_doc = {
                                    "id": context["id"],
                                    "candidates": candidates,
                                    "content": [
                                        {"sentence": s.text,
                                         "tokens": [t.text for t in s]} for s in
                                        spacy_doc.sents if len(sent) > 0 and len(sent.text.strip()) > 0
                                    ]
                                }
                                if field == 'document':
                                    f_out_doc.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
                                else:
                                    f_out_sum.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
                    f_out_doc.flush()
                    f_out_sum.flush()
    logger.info('Done')


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the train, test and validation JSON files for NarraSum."
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,

        help="The output data directory."
    )
    parser.add_argument(
        "--field",
        default=None,
        type=str,
        required=True,
        help="Data field to be processed, either 'document' or 'summary'"
    )
    parser.add_argument(
        "--spacy_model",
        default="en_core_web_sm",
        type=str,
        help="The spaCy model to use for dependency parsing, NER and lemmatization."
    )
    parser.add_argument(
        "--preprocessing_method",
        default="preprocess_narrasum_nltk_spacy",
        type=str,
        choices=["preprocess_narrasum_nltk_spacy", "preprocess_narrasum_spacy"],
        help="Which preprocessing function to use."
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    else:
        logger.warning("Overwriting existing output directory!")

    if args.preprocessing_method == "preprocess_narrasum_spacy":
        preprocess_narrasum_spacy(
            input_path=args.input_path,
            output_path=args.output_path,
            spacy_model=args.spacy_model
        )
    else:
        preprocess_narrasum_nltk_spacy(
            input_path=args.input_path,
            output_path=args.output_path,
            field=args.field,
            spacy_model=args.spacy_model
        )


if __name__ == "__main__":
    main()
