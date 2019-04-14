import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))

import argparse
import json
from collections import Counter

from helper import parse_path, valid_text_file_name_prefix

def read_text_file_update_counter(text_fp: Path, vocab_counter: Counter
                                  ) -> Counter:
    with text_fp.open('r')as text_file:
        for line in text_file:
            line = line.strip()
            if line:
                tokens = str.split(line)
                vocab_counter.update(tokens)
    return vocab_counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("text_fp", type=parse_path, 
                        help='Text file to convert to json')
    parser.add_argument("vocab_fp", type=parse_path, 
                        help='File Path to store the json data to')
    parser.add_argument("--file_prefix", type=str)
    parser.add_argument("--highest", action="store_true")
    args = parser.parse_args()

    vocab_counter = Counter()
    text_fp = args.text_fp
    file_prefix = args.file_prefix

    if text_fp.is_dir():
        text_dir = text_fp
        for text_fp in text_dir.iterdir():
            if not valid_text_file_name_prefix(file_prefix, text_fp.name):
                print(f'Not including {text_fp.name}')
                continue
            print(f'including {text_fp.name}')
            vocab_counter = read_text_file_update_counter(text_fp, vocab_counter)

    else:
        if not valid_text_file_name_prefix(file_prefix, text_fp.name):
            raise FileNotFoundError(f'The file prefix `{file_prefix}` given'
                                    ' does not match the given file name '
                                    f'`{text_fp.name}`')
        vocab_counter = read_text_file_update_counter(text_fp, vocab_counter)
        
    
    filtered_counter = Counter()
    not_in = []
    with args.vocab_fp.open('r') as vocab_file:
        external_vocab = set(json.load(vocab_file))
        for word in external_vocab:
            if word in vocab_counter:
                filtered_counter[word] = vocab_counter[word]
            else:
                not_in.append(word)
    if args.highest:
        highest_word, count = sorted(filtered_counter.items(), key=lambda x: x[1], reverse=True)[0]
        print(f'Highest word {highest_word} and count {count}')
    else:
        lowest_word, count = sorted(filtered_counter.items(), key=lambda x: x[1], reverse=False)[0]
        print(f'Highest word {lowest_word} and count {count}')