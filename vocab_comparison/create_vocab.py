import argparse
from pathlib import Path
from typing import Callable, List, Dict, Set
import json
import re

import spacy
from spacy.cli.download import download as spacy_download
from spacy.cli import link
from spacy.util import get_package_path
from spacy.language import Language as SpacyModelType


LOADED_SPACY_MODELS: Dict[str, SpacyModelType] = {}

def _get_spacy_model(language: str) -> SpacyModelType:
    """
    To avoid laoding lots of spacy models the model specific to a language 
    is loaded and saved within a Global dictionary.
    This has been mainly taken from the `AllenNLP package <https://github.
    com/allenai/allennlp/blob/master/allennlp/common/util.py>`_
    :param language: Language of the SpaCy model to load.
    :returns: The relevant SpaCy model.
    """
    if language not in LOADED_SPACY_MODELS:
        disable = ['vectors', 'textcat', 'tagger', 'parser', 'ner']
        try:
            spacy_model = spacy.load(language, disable=disable)
        except OSError:
            print(f"Spacy models '{language}' not found.  Downloading and installing.")
            spacy_download(language)
            package_path = get_package_path(language)
            spacy_model = spacy.load(language, disable=disable)
            link(language, language, model_path=package_path)
        LOADED_SPACY_MODELS[language] = spacy_model
    return LOADED_SPACY_MODELS[language]

def spacy_tokeniser(text: str) -> List[str]:
    '''
    `SpaCy tokeniser <https://spacy.io/>`_
    Assumes the language to be English.
    :param text: A string to be tokenised.
    :returns: A list of tokens where each token is a String.
    '''
    spacy_model = _get_spacy_model('en')

    spacy_document = spacy_model(text)
    tokens = []
    for token in spacy_document:
        if not token.is_space:
            tokens.append(token.text)
    return tokens


def get_tokeniser(tokeniser_name: str) -> Callable[[str], List[str]]:
    '''
    Given a tokeniser name it will return that tokeniser function.
    '''
    if tokeniser_name == 'spacy':
        print('Using the spaCy tokeniser')
        return spacy_tokeniser
    elif tokeniser_name == 'whitespace':
        print('Using the WhiteSpace tokeniser')
        return str.split
    else:
        raise ValueError(f'Not a recognised tokeniser name: {tokeniser_name}')
        
def create_vocab(text_fp: Path, tokeniser: Callable[[str], List[str]]
                 ) -> Set[str]:
    '''
    Given a file that contains just text on each new line it will tokenise that 
    text and return a Set of all the unique words (tokens).
    '''
    vocab = set()
    with text_fp.open('r') as text_file:
        for line in text_file:
            line = line.strip()
            if line:
                tokens = tokeniser(line)
                for token in tokens:
                    vocab.add(token)
    return vocab

def valid_text_file_name_prefix(prefix_pattern: str, text_file_name: str
                                ) -> bool:
    if prefix_pattern:
        if re.search(rf'^{prefix_pattern}', text_file_name) is None:
            return False
    return True

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    text_fp_help = 'File Path to the text file that the vocabulary is built from'
    vocab_fp_help = 'File Path to store the created vocabulary'
    tokeniser_help = 'Which tokeniser to use, default is spaCy'
    from_dir_of_files_help = 'Whether or not the text File Path given is a '\
                             'directory of text files that the vocab is to be '\
                             'created from'
    file_prefix_help = 'Optional filter that ensures a file is only read if '\
                       'this is in the start of the file name e.g. `news` for'\
                       ' files like `news.en-00001-of-00100`'
    parser = argparse.ArgumentParser()
    parser.add_argument("text_fp", type=parse_path, help=text_fp_help)
    parser.add_argument("vocab_fp", type=parse_path, help=vocab_fp_help)
    parser.add_argument("tokeniser", type=str, choices=['spacy', 'whitespace'], 
                        default='spacy', help=tokeniser_help)
    parser.add_argument("--from_dir_of_files", help=from_dir_of_files_help,
                        action='store_true')
    parser.add_argument("--file_prefix", help=file_prefix_help, type=str)
    args = parser.parse_args()

    text_fp = args.text_fp
    vocab_fp = args.vocab_fp
    tokeniser = get_tokeniser(args.tokeniser)

    file_prefix = args.file_prefix
    if args.from_dir_of_files:
        vocab = set()
        if not text_fp.is_dir():
            raise ValueError(f'Expect the text_fp `{text_fp}` argument to be a'
                             ' folder.')
        text_dir: Path = text_fp
        for text_fp in text_dir.iterdir():
            if not valid_text_file_name_prefix(file_prefix, text_fp.name):
                print(f'Not including {text_fp.name}')
                continue
            print(f'including {text_fp.name}')
            vocab = vocab.union(list(create_vocab(text_fp, tokeniser)))
        vocab = list(vocab)
    else:
        if not valid_text_file_name_prefix(file_prefix, text_fp.name):
            raise FileNotFoundError(f'The file prefix `{file_prefix}` given'
                                    ' does not match the given file name '
                                    f'`{text_fp.name}`')
        vocab = list(create_vocab(text_fp, tokeniser))
    with vocab_fp.open('w+') as vocab_file:
        json.dump(vocab, vocab_file)
    print(f'Vocabulary has been created and written to {vocab_fp}')
