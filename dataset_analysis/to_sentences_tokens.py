import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))
if str(Path(__file__, '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..').resolve()))

import argparse
import json
from typing import Iterable, List, Optional, Callable

from spacy.lang.en import English
from spacy.language import Language as SpacyModelType

import helper
from data_analysis_helper import yelp_text_generator, text_generator_func_mapper

SPACY_MODEL = {}

def get_sentence_spitter() -> SpacyModelType:
    if 'english' not in SPACY_MODEL:
        nlp = English()  # just the language with no model
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)
        SPACY_MODEL['english'] = nlp
    return SPACY_MODEL['english']

def sentence_token_generator(text: str) -> Iterable[str]:
    '''
    Given some text that can contain multiple sentences it breaks the document
    into sentences, tokenises the sentence and yields each sentence where the 
    original tokenisation can be found by splitting on whitespace.

    Tokenisation and sentence splitting done by spacy.
    '''
    spacy_pipeline = get_sentence_spitter()
    spacy_annotations = spacy_pipeline(text)
    try:
        for sentence in spacy_annotations.sents:
            tokens = []
            for token in sentence:
                if not token.is_space:
                    tokens.append(token.text)
            yield ' '.join(tokens)
    except:
        print(f'Text that Spacy cannot parse {text}\nThis text will not be included.')

def split_save_reviews(text_generator_func: Callable[[Path], Iterable[str]], 
                       review_data_dir: Path, 
                       split_names: Optional[List[str]] = None) -> None:
    split_names = split_names or ['train.json', 'val.json', 'test.json']
    save_split_names = ['split_train.txt', 'split_val.txt', 'split_test.txt']

    for split_name, save_split_name in zip(split_names, save_split_names):
        split_fp = Path(review_data_dir, split_name).resolve()
        save_split_fp = Path(review_data_dir, save_split_name).resolve()

        count = 0
        with save_split_fp.open('w+') as save_split_file:
            for review_text in text_generator_func(split_fp):
                for sentence in sentence_token_generator(review_text):
                    if count != 0:
                        sentence = f'\n{sentence}'
                    save_split_file.write(sentence)
                    count += 1
        print(f'{split_fp} has been sentence split and tokenised and the corresponding'
              f' sentences have been written to the following file {save_split_fp}')
        print(f'Number of sentences created {count}')

if __name__ == '__main__':
    '''
    Given a folder that contains a datasets splits e.g. Yelp dataset and it's 
    train, val, and test splits, it will convert those datasets into files 
    that only contain sentence on each new line where each sentence has been 
    tokenised. 
    '''

    data_dir_help = "Directory that contains the train, val, and test splits,"\
                    " this will also be the directory where the new data "\
                    "will be stored."

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=helper.parse_path, 
                        help=data_dir_help)
    parser.add_argument("dataset_name", help="Name of the dataset e.g. yelp", 
                        choices=["yelp", "amazon"], type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        raise ValueError(f'Data directory {data_dir} is not a directory.')
    text_generator_func = text_generator_func_mapper(args.dataset_name)

    split_save_reviews(text_generator_func, data_dir)