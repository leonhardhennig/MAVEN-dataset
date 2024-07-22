#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 18.06.24
@author: leonhard.hennig@dfki.de
"""
import json
import os
import re

import nltk
import uuid
import spacy
import argparse
import logging
import stanza
import pandas as pd

from tqdm import tqdm
from itertools import islice

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def preprocess_narrasum_nltk_spacy(input_path, output_path, field, spacy_model="en_core_web_sm"):
    # Clément's version

    # spacy mainly for PoS tagging
    if spacy_model == "en_core_web_trf":
        spacy.prefer_gpu()
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
        yield docs


def convert_to_regular_spaces(text):
    # Define a regex pattern that includes all the special Unicode space characters.
    unicode_spaces = r'[\u0020\u00A0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u202F\u205F\u3000]'
    # Replace all occurrences of these characters with a regular space.
    return re.sub(unicode_spaces, ' ', text)


def preprocess_narrasum_spacy(input_path, output_path, field, spacy_model="en_core_web_trf", batch_size=100):
    # Leo's version (but separate processing for document & summary) and a bug fix: spaCy for everything
    if spacy_model == "en_core_web_trf":
        spacy.prefer_gpu()
    try:
        nlp = spacy.load(spacy_model, disable=["attribute_ruler, lemmatizer, ner"])
    except OSError:
        from spacy.cli.download import download as spacy_download
        spacy_download(spacy_model)
        nlp = spacy.load(spacy_model, disable=["attribute_ruler, lemmatizer, ner"])

    for split in ["train", "test", "validation"]:
        logger.info(f'Preprocessing {split}')
        input_file = f"{input_path}/{split}.json"
        output_file = f"{output_path}/{split}_{field}.jsonl"
        with open(output_file, 'w') as f_out, open(input_file, 'r') as f_in:
            for doc_batch in tqdm(get_next_batch(f_in, batch_size=batch_size)):
                doc_batch_tuples = [(convert_to_regular_spaces(doc[field]), doc) for doc in doc_batch]
                spacy_doc_tuples = nlp.pipe(doc_batch_tuples, batch_size=batch_size, as_tuples=True)
                write_buffer = []
                for spacy_doc, context in spacy_doc_tuples:
                    candidates = []
                    content = []
                    sent_id = 0
                    for sent in spacy_doc.sents:
                        if len(sent) == 0 or len(sent.text.strip()) == 0:
                            logger.warning(f"Empty {sent.text=} after sentence {sent_id=} in doc['id']={context['id']}")
                            continue
                        new_sent = {
                            "sentence": sent.text,
                            "tokens": []
                        }
                        for j, tok in enumerate(sent):
                            new_sent["tokens"].append(tok.text)
                            if tok.pos_ in ["NOUN", "VERB", "ADJ"]:
                                dic = {
                                    "trigger-word": tok.text,
                                    "sent_id": sent_id,
                                    "offset": [j, j + 1],
                                    "id": str(uuid.uuid4()).replace("-", "")
                                }
                                candidates.append(dic)
                        content.append(new_sent)
                        sent_id += 1
                    new_doc = {
                        "id": context["id"],
                        "candidates": candidates,
                        "content": content
                    }
                    write_buffer.append(json.dumps(new_doc, ensure_ascii=False) + "\n")
                f_out.writelines(write_buffer)
            f_out.flush()
    logger.info('Done')


def preprocess_narrasum_stanza(input_path, output_path, field, batch_size=100):
    # Use stanza's neural pipeline for preprocessing: slower, but more accurate than spacy
    stanza.download("en", processors="tokenize, pos")
    nlp = stanza.Pipeline("en", processors="tokenize, pos")
    for split in ["train", "test", "validation"]:
        logger.info(f"Preprocessing {split}")
        input_file = f"{input_path}/{split}.json"
        output_file = f"{output_path}/{split}_{field}.jsonl"
        with open(output_file, 'w') as f_out, open(input_file, 'r') as f_in:
            for doc_batch in tqdm(get_next_batch(f_in, batch_size=batch_size)):
                texts = [convert_to_regular_spaces(doc[field]) for doc in doc_batch]
                processed_doc_batch = nlp.bulk_process(texts)
                for processed_doc, doc in zip(processed_doc_batch, doc_batch):
                    candidates = []
                    content = []
                    sent_id = 0
                    for sent in processed_doc.sentences:
                        if (len(sent.text.strip()) == 0 or len(sent.words) == 0 or
                                all(len(w.text.strip()) == 0 for w in sent.words)):
                            logger.warning(f"Empty {sent.text=} after sentence {sent_id=} in {doc['id']=}")
                            continue
                        new_sent = {
                            "sentence": sent.text,
                            "tokens": []
                        }
                        for j, tok in enumerate(sent.words):
                            # alternatively iterate over sent.tokens and access pos tag via token.words[0].upos
                            token_text = tok.text
                            new_sent["tokens"].append(token_text)
                            pos_tag = tok.upos
                            if pos_tag in ["NOUN", "VERB", "ADJ"]:
                                dic = {
                                    "trigger-word": token_text,
                                    "sent_id": sent_id,
                                    "offset": [j, j + 1],
                                    "id": str(uuid.uuid4()).replace("-", "")
                                }
                                candidates.append(dic)
                        content.append(new_sent)
                        sent_id += 1
                    new_doc = {
                        "id": doc["id"],
                        "candidates": candidates,
                        "content": content
                    }
                    f_out.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
    logger.info("Done")


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
        help="The spaCy model to use for (sentence segmentation,) tokenization, part of speech tagging."
    )
    parser.add_argument(
        "--preprocessing_method",
        default="preprocess_narrasum_nltk_spacy",
        type=str,
        choices=["preprocess_narrasum_nltk_spacy", "preprocess_narrasum_spacy", "preprocess_narrasum_stanza"],
        help="Which preprocessing function to use."
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="Batch size for processing documents with spacy/stanza."
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    else:
        logger.warning("Overwriting existing output directory!")

    logger.info(f"Preprocessing NarraSum data from {args.input_path} to {args.output_path} using "
                f"{args.preprocessing_method}")
    if args.preprocessing_method == "preprocess_narrasum_spacy":
        preprocess_narrasum_spacy(
            input_path=args.input_path,
            output_path=args.output_path,
            field=args.field,
            spacy_model=args.spacy_model,
            batch_size=args.batch_size
        )
    elif args.preprocessing_method == "preprocess_narrasum_stanza":
        preprocess_narrasum_stanza(
            input_path=args.input_path,
            output_path=args.output_path,
            field=args.field,
            batch_size=args.batch_size
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
