import argparse
from typing import Callable, List, Iterable, Tuple, Union, Optional
from pathlib import Path
import multiprocessing

from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.phrases import Phrases
from gensim.models import Word2Vec

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

class TokenIter():
    def __init__(self, token_fp: Path, lower: bool = True) -> None:
        '''
        :param token_fp: Path to a file that contains a line of text that 
                            has already been tokenized.
        :param lower: Whether to lower case the text
        '''
        self.token_fp = token_fp
        self.lower = lower

    def __iter__(self) -> Iterable[List[str]]:
        '''
        Iterates over the file path given within the constructor and generates
        the tokens of each json object per iteration.
        :return: Tokens
        '''
        with self.token_fp.open('r') as token_file:
            for line in token_file:
                if line.strip():
                    if self.lower:
                        line = line.lower()
                    tokens = line.split()
                    yield tokens

def creating_embeddings(text_file: Path,
                        embedding_fp: Path,
                        embedding_class: BaseWordEmbeddingsModel,
                        lower: bool = True, 
                        n_grams: Optional[int] = None,
                        **embedding_kwargs) -> Path:
    '''
    '''
    if embedding_fp.is_file():
        return embedding_fp
    # Create the id, token file if it does not exist
    text_file.parent.mkdir(parents=True, exist_ok=True)
    id_token_fp = text_file.with_name(f'{text_file.name}')
    if lower:
        id_token_fp = text_file.with_name(f'lower {text_file.name}')
    if not id_token_fp.is_file():
        with id_token_fp.open('w+') as id_token_file:
            for index, tokens in enumerate(TokenIter(text_file, lower=lower)):
                if index != 0:
                    id_token_file.write('\n')
                if lower:
                    tokens = [token.lower() for token in tokens]
                id_token_file.write(' '.join(tokens))
    token_generator = TokenIter(id_token_fp, lower=lower)
    # Adds the phrase based logic
    if n_grams is not None:
        all_tokens = TokenIter(id_token_fp, lower=lower)
        for n_gram in range(2, n_grams + 1):
            phrase_fp = id_token_fp.name.split('.')[0]
            phrase_fp = id_token_fp.with_name(f'{phrase_fp} phrases_{n_gram}')
            phrase_fp = phrase_fp.with_suffix('.npy')
            new_tokens_fp = phrase_fp.with_suffix('.txt')
            if phrase_fp.exists() and new_tokens_fp.exists():
                all_tokens = TokenIter(id_token_fp, lower=lower)
                continue
            phrases = Phrases(token_generator)
            with new_tokens_fp.open('w+') as new_tokens_file:
                for index, tokens in enumerate(all_tokens):
                    if index != 0:
                        new_tokens_file.write('\n')
                    phrase_tokens = phrases[tokens]
                    new_tokens_file.write(' '.join(phrase_tokens))
            phrases.save(str(phrase_fp.resolve()))
            all_tokens = TokenIter(new_tokens_fp, lower=lower)
        token_generator = all_tokens

    embedding_model = embedding_class(token_generator, **embedding_kwargs)
    embedding_fp.parent.mkdir(parents=True, exist_ok=True)
    embedding_path = str(embedding_fp.resolve())
    embedding_model.save(embedding_path)
    return embedding_fp

if __name__ == '__main__':
    '''
    It uses Skip Gram
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data", type=parse_path,
                        help='File path to the training data.')
    parser.add_argument("n_grams", type=int,
                        help='The number of n-grams the word embedding has to go to')
    parser.add_argument("embedding_file", type=parse_path, 
                        help='File path to save the embedding to')
    args = parser.parse_args()

    embedding_dims = 300
    num_workers = multiprocessing.cpu_count() - 2
    n_grams = args.n_grams
    if n_grams == 1:
        n_grams = None
    text_fp = args.training_data
    embedding_fp = args.embedding_file

    creating_embeddings(text_fp, embedding_fp, embedding_class=Word2Vec, 
                        n_grams=n_grams, lower=True, size=embedding_dims, 
                        workers=num_workers, sg=1)