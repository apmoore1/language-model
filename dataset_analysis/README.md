# Data Staistics of the language model datasets
## 1 Billion word corpus
As the original data the [Transformer ELMo model](https://allennlp.org/elmo) was trained on shuffled sentence from the one billion word corpus we wanted to see what the data statistics of this corpus was e.g. Max, Mean, Minimum sentence lengths etc therefore this can be seen by running the following command:
``` bash
python dataset_analysis/data_stats.py ../1-billion-word-language-modeling-benchmark-r13output/ billion_word_corpus --sentence_length_distribution ./images/sentence_distributions/1_billion_word_corpus.png
```
As we can seen the `max`, `mean`, and `min` token counts per sentence are 2818, 25.36, and 1 respectively. We can also see that very few sentences only contain one token (182) and further more very few contain only 2 tokens (19516). We can also see that the number of tokens that occur at least three times for the combined TRAIN and TEST set is 797,965 showing that the diversity in tokens is quiet large.

NOTE: `./images/sentence_distributions/1_billion_word_corpus.png` is just a graph of the sentence distribution.

## Yelp sentences

``` bash
python dataset_analysis/data_stats.py ../yelp/splits yelp_sentences --sentence_length_distribution ./images/sentence_distributions/yelp.png
```
As we can seen the `max`, `mean`, and `min` token counts per sentence are 937, 15.54, and 1 respectively. The fact that the sentences are shorter in general compared to the 1 billion word corpus could be the domain or the spacy sentence splitter. We can also see that the diversity of the tokens is smaller compared to the 1 billion word corpus as the number of tokens that occur at least three times in all of the data splits is only 260,150 compared to 797,965. Lastly we can also see that there are a lot more sentences with only 1 or 2 words in them of which I think this is most likely due to the sentence splitter and some what reviews in general e.g. review that just say good, great etc.