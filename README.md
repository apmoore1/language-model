When we are splitting the Yelp dataset we are going to keep 8% of the reviews for test and validation sets and 84% for training which is very similar number of articles (8.3%) that were used for the [wikiText2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)


from what I gather in the hugginface version they add <eos> at the start as the starting symbol and in the original version they do not but they both share that <eos> happens at the end of a line even if the end of the line happens multiple times thus this can occur <eos> <eos> this is only for the wt103 dataset also I think the wt103 dataset is the only one where the model only knows the words in the training data and not in any of the other datasets.


From the paper the transformer XL states that there are two problems:
1. With the traditional transformer you have to have a fixed length context window due to memory and compute reasons, thus with this you ignore the extended context from the other related contexts which is the context fragmentation problem
2. Having this fixed context and segements will not allow you to learn long term dependency between words that are across segements, this is probably more important in story books/novels.


We are going to use the pre-trained model from wikiText-103 as it has long term dependencies, 103M training tokens, 28K articles, average length of 3.6K per article. 



I was wondering what the license terms were for the Yelp dataset and then also the amazon dataset and then I was thinking for Twitter as well.

The Yelp data license I think this is the key part in section 3:
"Term to use, access, and create derivative works of the Data in electronic form for academic purposes only." I think this could be an interesting problem as a language model I suppose could generate an identical review but without knowing: "i.e. you may not publicly display any of the Data to any
third party, especially reviews and other user generated content, as this is a private data set
challenge and not a license to compete with or disparage with Yelp" I also think this part of the aggrement could be legally tricky with respect to publishing the model "rent, lease, sell, transfer, assign, or sublicense, any part of the Data" Section 5 also falls into this with respect to who would own the model as in the does section 5 just say that the only company that could have any rights over it is the Yelp company.


So from what I have gathered the reason why the test data in Wiki103 is <eos> is because that is the symbol that is applied for new line and the first token in the test data is a new line.


I think the next step would be to just split the yelp dataset into train, val and test based on previous work in the area on the size of train, val, and test.

The next step after that would be to look at the size of the dataset by the token size where we will split based on whitespace. Then I think we will look into which model to use as the TransformerXL from what I have just gathered only uses token level information and does not encode any characters which is problematic.

## How to run the Transformer ELMo model

command:


``` bash
allennlp evaluate --cuda-device -1 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 500], "max_instances_in_memory": 512, "batch_size": 128 }}}' transformer-elmo-2019.01.10.tar.gz 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100
```

## What is the vocab comparison folder
The vocab comparison folder contains script that will create a `vocab.json` file (or whatever you would like to call it) that will contain a list of all vocabulary terms given a corpus to create the vocabularly from and a tokeniser to `create/split` the words within the corpus.

To create a vocabulary file from the 1 Billion word held out corpus:
``` bash
python create_vocab.py 1-Billion-Held-Out-Corpus-File-Path Vocab-File-Path whitespace
python vocab_comparison/create_vocab.py ../1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100 ../vocab_1_billion_held_out whitespace

```
Given the location of the 1 Billion word held out corpus it will create the json vocabulary file at the given path using the whitespace tokeniser.
