import argparse
import json
from pathlib import Path

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_fp", type=parse_path, 
                        help='Text file to convert to json')
    parser.add_argument("vocab_json_fp", type=parse_path, 
                        help='File Path to store the json data to')
    args = parser.parse_args()

    vocab = set()
    with args.vocab_fp.open('r') as vocab_file:
        for line in vocab_file:
            line = line.strip()
            if line:
                vocab.add(line)
    with args.vocab_json_fp.open('w+') as vocab_json_file:
        vocab = list(vocab)
        json.dump(vocab, vocab_json_file)
    print('Vocab data is stored in json format at the following File Path '
          f'{args.vocab_json_fp}')