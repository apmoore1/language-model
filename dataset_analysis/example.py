import argparse
from pathlib import Path
from typing import List
import json

import spacy
from spacy.cli.download import download as spacy_download

def get_spacy_model(disable: List[str],
                    spacy_model_name: str = "en_core_web_sm"):
    from spacy.cli import link
    from spacy.util import get_package_path
    spacy_download(spacy_model_name)
    package_path = get_package_path(spacy_model_name)
    link(spacy_model_name, spacy_model_name, model_path=package_path)
    return spacy.load(spacy_model_name, disable=disable)


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_fp", type=parse_path)
    parser.add_argument("number_lines", type=int, default=100)
    parser.add_argument("write_fp", type=parse_path)
    args = parser.parse_args()

    disable = ['vectors', 'textcat', 'ner', 'tagger']
    try:
        spacy_model = spacy.load("en_core_web_sm", disable=disable)
    except OSError:
        spacy_model = get_spacy_model(spacy_model_name="en_core_web_sm", disable=disable)
    count = 0
    with args.write_fp.open('w+') as b_file:
        with args.file_fp.open('r') as a_file:
            for line in a_file:
                if count == args.number_lines:
                    break
                print(json.loads(line))
                text = json.loads(line)['text']
                doc = spacy_model(text)
                for sent in doc.sents:
                    sent = ' '.join([token.text for token in sent])
                    b_file.write(sent+'\n')
                count += 1
                b_file.write('\nEND\n')



