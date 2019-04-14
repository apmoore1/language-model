import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))

import argparse
from typing import Callable, List, Dict, Set
import json
import re

from helper import parse_path, get_tokeniser, valid_text_file_name_prefix

        
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
