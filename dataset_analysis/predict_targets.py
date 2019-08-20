import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Any, List

from target_extraction.allen import AllenNLPModel

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def token_iter(fp: Path) -> Iterable[Dict[str, List[str]]]:
    with fp.open('r') as _file:
        for line in _file:
            line = line.strip()
            if line:
                tokens = line.split()
                yield {'tokens': tokens, 'text': line}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file_path", type=parse_path, 
                        help='File Path to the TDSA model')
    parser.add_argument("model_param_path", type=parse_path, 
                        help='File Path to the TDSA model param file')
    parser.add_argument("subsampled_fp", type=parse_path,
                        help="File path to the subsampled sentences")
    parser.add_argument("targets_fp", type=parse_path,
                        help="File Path to the predicted targets")
    args = parser.parse_args()

    model = AllenNLPModel('ELMO model', args.model_param_path, 'target-tagger', 
                          args.model_file_path)
    model.load(cuda_device=0)
    count = 0
    overall_count = 0
    from time import time
    start_time = time()
    first_line = True
    with args.targets_fp.open('w+') as target_file:
        token_stream = token_iter(args.subsampled_fp)
        token_stream_tokens = token_iter(args.subsampled_fp)
        try:
            for prediction in model.predict_sequences(token_stream):
                labels = prediction['sequence_labels']
                confidence = prediction['confidence']
                tokens = next(token_stream_tokens)['tokens']

                assert len(tokens) == len(labels), f'ASSERT: {tokens} {labels}'
                assert len(tokens) == len(confidence), f'ASSERT: {tokens} {confidence}'
                text = ' '.join(tokens)
                output_dict = {'tokens': tokens, 'text': text, 
                            'predicted_sequence_labels': labels, 
                            'label_confidence': confidence}
                output_dict = json.dumps(output_dict)
                if first_line:
                    first_line = False
                    target_file.write(output_dict)
                else:
                    target_file.write(f'\n{output_dict}')

                if count == 10000:
                    count = 0
                    print(f'Processing time: {time() - start_time}')
                    print(f'Number done so far: {overall_count}')
                count += 1
                overall_count += 1
        except:
            print(f'count so far {overall_count}')
            print(f'Failed on these tokens {next(token_stream_tokens)}')

