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

# Getting the data and models
## Target Dependent Sentiment Analysis (TDSA) data
### Getting the data and converting it into Train, Validation, and Test sets

Download the following datasets:

1. SemEval Restaurant and Laptop 2014 [1] [train](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and [test] and put the `Laptop_Train_v2.xml` and `Restaurants_Train_v2.xml` training files into the following directory `./tdsa_data` and do the same for the test files (`Laptops_Test_Gold.xml` and `Restaurants_Test_Gold.xml`)
2. Election dataset [2] from the following [link](https://ndownloader.figshare.com/articles/4479563/versions/1) and extract all of the data into the following folder `./tdsa_data/election`, the `election` folder should now contain the following files `annotations.tar.gz`, `test_id.txt`, `train_id.txt`, and `tweets.tar.gz`. Extract both the `annotations.tar.gz` and the `tweets.tar.gz` files.

Then run the following command to create the relevant and determinstic train, validaion, and test splits of which these will be stored in the following directory `./tdsa_data/splits`:

``` bash
python ./tdsa_data/generate_datasets.py ./tdsa_data ./tdsa_data/splits
```

This should create all of the splits that we will use throughout the normal experiments that are the baseline values for all of our augmentation experiments. This will also print out some statistics for each of the splits to ensure that they are relatively similar.

## 1 Billion word corpus Transformer ELMo language model
The [Transformer ELMo model](https://allennlp.org/elmo) that came from the following [paper](https://aclweb.org/anthology/D18-1179) was trained on the 1 Billion word corpus that can be downloaded from [here](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz) and a good tutorial about the model can be found [here](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/training_transformer_elmo.md). As the downloaded 1 billion corpus has already been tokenised and the Transformer ELMo model that has been downloaded comes with a fixed output vocabulary, we show that the vocabularly comes from only the training corpus and from words that have a frequency of at least 3 in the training corpus. We also find that not all the words in the test corpus are in the training corpus of the 1 billion word corpus.

NOTE: The model's vocabularly comes from un-tarring `transformer-elmo-2019.01.10.tar.gz` the model Transformer ELMo model download and going into the vocab folder.
NOTE: It is required to un-tar the `transformer-elmo-2019.01.10.tar.gz` as the vocabulary is required and the weights therefore we assume you have un-tared it into the following folder `../transformer_unpacked`

To see how we discovered how the Transformer ELMo model output vocabularly was discovered and how this vocabularly MORE IMPORTANTLY overlaps with the TDSA data and the TDSA target words look at the following [mark down file](./vocab_comparison/README.md)



## Yelp Data and filtering it

The [Yelp dataset](https://www.yelp.com/dataset) was accessed on the week starting the 1st of April 2019, this is important as Yelp releases a new dataset every year. We only used reviews that review businesses from the following categories `restaurants` `restaurant` `restaurants,` this was to ensure that the domain of the reviews were similar to the restaurant review domain (some reviews are about hospitals etc). To filter the reviews we ran the following command:
``` bash
python dataset_analysis/filter_businesses_by_category.py YELP_REVIEW_DIR/business.json business_filter_ids.json 'restaurants restaurant restaurants,'
```
Where `YELP_REVIEW_DIR` is the directory that contains the downloaded [Yelp dataset](https://www.yelp.com/dataset), `business_filter_ids.json` is the json file you want to store the filtered business ids that will only contain business ids that have come from the restaurant domain based on the 3rd argument `restaurants restaurant restaurants,` which specifies the categories the Yelp business must be within to be allowed in the `business_filter_ids.json` file.

### Converting the Yelp data into sentences and tokens.
Once we have the filtered data we are only interested in the text of the data as we want to fine tune the [Transformer ELMo model](https://allennlp.org/elmo) from the news corpus data it was original trained on to restaurant reviews. This is because restaurant reviews use a slightly different vocabularly (for examples see [this](./vocab_comparison/README.md)) and more than likely a different language style to news data. 

To convert the split Yelp reviews into split Yelp data that has been tokenised by spacy and only contains one sentence per line (same format as 1 Billion word corpus) run the following command:
``` bash
python dataset_analysis/to_sentences_tokens.py ../yelp/splits yelp
```
This will create three more files within the `../yelp/splits` directory; `split_train.txt`, `split_val.txt`, `split_test.txt`. This dataset will now be called **yelp sentences**

## Yelp sentence filtering based on sentence length and the similarity to the TDSA dataset
### Filtering

Based on the [data statistics](./dataset_analysis/README.md) we are going to further filter the Yelp sentences dataset so that it only includes sentences that are at least 3 tokens long. We will also restrict the maximum sentence length to 40 as there are so few review sentences greater than this (2.48%). To do this run the following command:

``` bash
python dataset_analysis/filter_by_sentence_length.py ../yelp/splits yelp_sentences 3 40
```

This will create three more files within the `../yelp/splits` directory; `filtered_split_train.txt`, `filtered_split_val.txt`, and `filtered_split_test.txt`. These train, validation, and test splits are the final dataset that we will use to fine tune the Transformer ELMo model on for the restaurant review domain. Now we can re-run the data statistics to see the proportions and distribution of token and sentence length frequencies of the new filtered yelp training, validation, and test data:
```
python dataset_analysis/data_stats.py ../yelp/splits/filtered_split_train.txt yelp_sentences --sentence_length_distribution ./images/sentence_distributions/yelp_filtered_training.png
python dataset_analysis/data_stats.py ../yelp/splits/filtered_split_val.txt yelp_sentences --sentence_length_distribution ./images/sentence_distributions/yelp_filtered_validation.png
python dataset_analysis/data_stats.py ../yelp/splits/filtered_split_test.txt yelp_sentences --sentence_length_distribution ./images/sentence_distributions/yelp_filtered_test.png
```
We find that the:
1. Training set has a mean sentence length of 14.78 (8.02) with 27,286,698 sentences and 218,903 tokens that occur at least 3 times. Distribution of the sentence lengths can be found [here](./images/sentence_distributions/yelp_filtered_training.png).
2. Validation set has a mean sentence length of 14.78 (8.01) with 2,606,292 sentences and 69,106 tokens that occur at least 3 times. Distribution of the sentence lengths can be found [here](./images/sentence_distributions/yelp_filtered_validation.png).
3. Test set has a mean sentence length of 14.77 (8.01) with 2,602,337 sentences and 68,588 tokens that occur at least 3 times. Distribution of the sentence lengths can be found [here](./images/sentence_distributions/yelp_filtered_test.png).
As we can see the sentence lengths and standard devations are very similar across the splits. 

### Similarity of the Yelp training data to the TDSA Restaurant data.
We want to compare the words that are in the Yelp Training data to those TDSA restaurant dataset. Therefore we are going to follow similar steps to those used in analysising the vocabulary of the Transformer ELMo and the TDSA data which can be found [here](./vocab_comparison/README.md). First we need to create a vocabulary for the Yelp Training data:
``` bash
python vocab_comparison/create_vocab.py ../yelp/splits/filtered_split_train.txt ../vocab_test_files/yelp_filtered_train.json whitespace
```
Now the vocabulary is created and assuming you have done the steps in [here](./vocab_comparison/README.md) we want to compare the restaurant TDSA
``` bash
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/restaurant_tdsa.json ../vocab_test_files/yelp_filtered_train.json ../vocab_test_files/tdsa_diff_between_yelp_train_and_restaurant.txt --not_symmetric
```
We find that there are 104 words that are not in the Yelp restaurant train dataset but are in the TDSA dataset, now lets look at difference in target words specifically:
``` bash
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/restaurant_target_tdsa.json ../vocab_test_files/yelp_filtered_train.json ../vocab_test_files/tdsa_diff_between_yelp_train_and_restaurant_targets.txt --not_symmetric
python vocab_comparison/targets_affected.py restaurant ../vocab_test_files/tdsa_diff_between_yelp_train_and_restaurant_targets.txt spacy tdsa_data/splits/
python vocab_comparison/targets_affected.py restaurant ../vocab_test_files/tdsa_diff_between_yelp_train_and_restaurant_targets.txt spacy tdsa_data/splits/ --unique
```
We find that there are 16 target words not in the Yelp training data and that these 16 target words affect only 16 samples out of the 4722 samples in the whole of the restaurant TDSA (train, validation, and test sets). Examples of these words:
``` python
['capex', 'AT MOSHPHERE', 'Guacamole+shrimp appetizer', 'clams oreganta', 'yellowfun tuna'] 
```

#### With the models actual vocabulary
``` bash
python vocab_comparison/txt_to_json.py ../yelp_lm_vocab/tokens.txt ../vocab_test_files/yelp_train_model.json 
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/restaurant_tdsa.json ../vocab_test_files/yelp_train_model.json ../vocab_test_files/tdsa_diff_between_yelp_train_model_and_restaurant.txt --not_symmetric
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/restaurant_target_tdsa.json ../vocab_test_files/yelp_train_model.json ../vocab_test_files/tdsa_diff_between_yelp_train_model_and_restaurant_targets.txt --not_symmetric
python vocab_comparison/targets_affected.py restaurant ../vocab_test_files/tdsa_diff_between_yelp_train_model_and_restaurant_targets.txt spacy tdsa_data/splits/
python vocab_comparison/targets_affected.py restaurant ../vocab_test_files/tdsa_diff_between_yelp_train_model_and_restaurant_targets.txt spacy tdsa_data/splits/ --unique
```
We find that there are 156 words that are not in the Yelp training vocabulary but are in the TDSA dataset of which 25 of these are target words that affect 26 targets and 26 samples of he 4722 samples across training, validation, and test sets.

## Amazon data
The amazon electronics data can be downloaded from [here](http://jmcauley.ucsd.edu/data/amazon/) we used the 5 core 1,689,188 reviews dataset. Once downloaded and un-compresed this is then put into it's own directory at `../amazon` and therefore the reviews can be accessed from the following path `../amazon/Electronics_5.json` (downloaded on the 2nd of May).
``` bash
python dataset_analysis/create_train_val_test.py ../amazon/Electronics_5.json ../amazon/ amazon
```
Number of training reviews 1418916(0.8399988633591998%)
Number of validation reviews 135136(0.0800005683204001%)
Number of test reviews 135136(0.0800005683204001%)
``` bash
python dataset_analysis/to_sentences_tokens.py ../amazon amazon
```
Based on the [data statistics](./dataset_analysis/README.md) we are going to further filter the Yelp sentences dataset so that it only includes sentences that are at least 3 tokens long. We will also restrict the maximum sentence length to 50 as there are so few review sentences greater than this (2.48%). To do this run the following command:
``` bash
python dataset_analysis/filter_by_sentence_length.py ../amazon yelp_sentences 3 50
```

```
python dataset_analysis/data_stats.py ../amazon/filtered_split_train.txt yelp_sentences --sentence_length_distribution ./images/sentence_distributions/amazon_filtered_training.png
python dataset_analysis/data_stats.py ../amazon/filtered_split_val.txt yelp_sentences --sentence_length_distribution ./images/sentence_distributions/amazon_filtered_validation.png
python dataset_analysis/data_stats.py ../amazon/filtered_split_test.txt yelp_sentences --sentence_length_distribution ./images/sentence_distributions/amazon_filtered_test.png
```
We find that the:
1. Training set has a mean sentence length of 18.06 (9.81) with 9,580,995 sentences and 182,900 tokens that occur at least 3 times. Distribution of the sentence lengths can be found [here](./images/sentence_distributions/yelp_filtered_training.png).
2. Validation set has a mean sentence length of 18.03 (9.81) with 907,343 sentences and 47,255 tokens that occur at least 3 times. Distribution of the sentence lengths can be found [here](./images/sentence_distributions/yelp_filtered_validation.png).
3. Test set has a mean sentence length of 18.07 (9.83) with 905,555 sentences and 47,178 tokens that occur at least 3 times. Distribution of the sentence lengths can be found [here](./images/sentence_distributions/yelp_filtered_test.png).
As we can see the sentence lengths and standard devations are very similar across the splits. 

## How to run the Transformer ELMo model

command:


``` bash
allennlp evaluate --cuda-device -0 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 500], "max_instances_in_memory": 512, "batch_size": 128 }}}' transformer-elmo-2019.01.10.tar.gz 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100
```
Did not find it any quicker to have more of the data in memory nor did the perplexity measure change.

## Fine tuning the Transformer ELMo model
In this section we show how you can fine tune the Transformer ELMo model to other domains and mediums.

### Amazon Review dataset
Assuming that you have created `filtered_split_train.txt`, `filtered_split_val.txt`, and `filtered_split_test.txt` from the previous sections, we will use these datasets to fine tune the model. First we must create a new output vocabulary for the Transformer ELMo model to do this easily use the following command:
``` bash
python fine_tune_lm/create_lm_vocab.py fine_tune_lm/training_configs/amazon_lm_vocab_create_config.json ../amazon_lm_vocab
```
Where `../amazon_lm_vocab` is a new directory that stores only the vocabulary files, of which the vocabulary that will be used can be found here `../amazon_lm_vocab/tokens.txt`.

#### Train model
To train the model run the following command (This will take a long time around 63 hours on a 1060 6GB GPU):
```
allennlp train fine_tune_lm/training_configs/amazon_lm_config.json -s ../amazon_language_model_save_large
```
Where `../amazon_language_model_save_large` is the directory that will save the language model to.

We got a perplexity score of 24.78 perplexity which is a loss of 3.21.

We did not do any quick test to see the difference of using a pre-trained model and not.

#### Evaluate the model
To evaluate the model we shall use the Amazon filterted test split which you should be able to find here if you followed the previous steps `../amazon/filtered_split_test.txt` and to evaluate you run the following command (This again will take around 1 hours 15 minutes on a 1060 6GB GPU):
``` bash
allennlp evaluate --cuda-device 0 ../amazon_language_model_save_large/model.tar.gz ../amazon/filtered_split_test.txt
```
We achieved a perplexity of 29.15 which is a loss of 3.37 (3.3725986638315453) which is close to the training loss. We now compare that to the non-fine tuned model which was trained on the 1 billion word corpus which came from news data, to do this run the following command:
``` bash
allennlp evaluate --cuda-device -0 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 300], "max_instances_in_memory": 16384, "batch_size": 512 }}}' ../transformer-elmo-2019.01.10.tar.gz ../amazon/filtered_split_test.txt
```
This will take longer to run (4 hours 30 minutes on a 1060 6GB GPU) as it has a much larger vocabulary therefore a much large softmax layer to compute within the neural network. We got a loss of 4.70 (4.7040290288591615) which is a perplexity score of 110.39. 

As we can see fine tunning to the dataset has made large difference with respect to the language model score.

### Yelp Restaurant Review dataset
Assuming that you have created `filtered_split_train.txt`, `filtered_split_val.txt`, and `filtered_split_test.txt` from the previous sections, we will use these datasets to fine tune the model. First we must create a new output vocabulary for the Transformer ELMo model to do this easily use the following command:
``` bash
python fine_tune_lm/create_lm_vocab.py fine_tune_lm/training_configs/yelp_lm_vocab_create_config.json ../yelp_lm_vocab
```
Where `../yelp_lm_vocab` is a new directory that stores only the vocabulary files, of which the vocabulary that will be used can be found here `../yelp_lm_vocab/tokens.txt`.

#### Train model
To train the model run the following command (This will take a long time around 49 hours on a 1060 6GB GPU):
```
allennlp train fine_tune_lm/training_configs/yelp_lm_config.json -s ../yelp_language_model_save_large
```
Where `../yelp_language_model_save_large` is the directory that will save the language model to.

We got a perplexity score of 24.78 perplexity which is a loss of 3.21.

We currently find that using the pre-trained model does not help at first but within 1 hour of training the perplexity decreases quicker suggesting that model finds it easier to learn more quickily through pre-training.

#### Evaluate the model
To evaluate the model we shall use the Yelp filterted test split which you should be able to find here if you followed the previous steps `../yelp/splits/filtered_split_test.txt` and to evaluate you run the following command (This again will take around 3 hours 25 minutes on a 1060 6GB GPU):
``` bash
allennlp evaluate --cuda-device 0 ../yelp_language_model_save_large/model.tar.gz ../yelp/splits/filtered_split_test.txt
```
We achieved a perplexity of 24.53 which is a loss of 3.20 (3.207513492262822) which is almost identical to the training loss. We now compare that to the non-fine tuned model which was trained on the 1 billion word corpus which came from news data, to do this run the following command:
``` bash
allennlp evaluate --cuda-device -0 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 300], "max_instances_in_memory": 16384, "batch_size": 512 }}}' ../transformer-elmo-2019.01.10.tar.gz ../yelp/splits/filtered_split_test.txt
```
This will take longer to run (10 hours 45 minutes on a 1060 6GB GPU) as it has a much larger vocabulary therefore a much large softmax layer to compute within the neural network. We got a loss of 4.61 (4.612278219667751) which is a perplexity score of 100.71. 

As we can see fine tunning to the dataset has made large difference with respect to the language model score.

Other suggestion for training better with a pre-trained model would be to use something like the [ULMFit model](https://arxiv.org/pdf/1801.06146.pdf) as currently we are using a learning rate schduler that is similar in warm up and decreasing but it does not care about the different layers i.e. does not freeze any of the layers at different epochs nor does it have a different learning rate for different layers all of this could be important for us. We have also not looked at the best learning rate which we could do through [fine learning rate](https://allenai.github.io/allennlp-docs/api/allennlp.commands.find_learning_rate.html?highlight=learning#module-allennlp.commands.find_learning_rate) which is based on the training data and batches. To find the number of parameter groups for the ULMFit model see [this](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py)

# Target Extraction 
Now that we have a training set for both Restaurants and Laptops we are going to sub-sample these into datasets that contain only 1 million sentences each, these randomly sub-sampled datasets can be create using the following script:
``` bash
python dataset_analysis/subsample_sentence.py ../yelp/splits/filtered_split_train.txt ../yelp/splits/sub_filtered_split_train.txt 1000000
python dataset_analysis/subsample_sentence.py ../amazon/filtered_split_train.txt ../amazon/sub_filtered_split_train.txt 1000000
```

Now we have done this we need to train a TargetExtraction model for both datasets. We are going to use the following datasets:
1. [SemEval 2014 task 4 Laptop domain (laptop)](http://alt.qcri.org/semeval2014/task4/). Of which the training data can be found [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and the test data [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/).
2. [SemEval 2014 task 4 Restaurant domain (restaurant_14)](http://alt.qcri.org/semeval2014/task4/). Of which the training and the test data can be found at the same place as the laptop dataset.

We shall also use the Domain Specific language models that we have created here as well as the Glove 300D general word embedding as the word representations for the Bi-LSTM with CRF decoding. First before training the models we need to split them into train, validation and test splits using the following script:
``` bash
python dataset_analysis/TDSA_create_splits.py ../original_target_datasets/semeval_2014/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Laptop_Train_v2.xml ../original_target_datasets/semeval_2014/ABSA_Gold_TestData/Laptops_Test_Gold.xml semeval_2014 ../original_target_datasets/semeval_2014/laptop_json/train.json ../original_target_datasets/semeval_2014/laptop_json/val.json ../original_target_datasets/semeval_2014/laptop_json/test.json
python dataset_analysis/TDSA_create_splits.py ../original_target_datasets/semeval_2014/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Restaurants_Train_v2.xml ../original_target_datasets/semeval_2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml semeval_2014 ../original_target_datasets/semeval_2014/restaurant_json/train.json ../original_target_datasets/semeval_2014/restaurant_json/val.json ../original_target_datasets/semeval_2014/restaurant_json/test.json
```

Then to train the models run the following:
``` bash
allennlp train TDSA_configs/ELMO_Laptop.jsonnet -s TDSA_Models/Laptop --include-package target_extraction
```
The Laptop dataset you should have an F1 score of 0.852 and 0.837 for test and validation sets.
The 

This should create two more models files both stored at the following ``

Now we want to create a file that will contain all of the predicted Targets, by running the following command:
``` bash
python dataset_analysis/predict_targets.py TDSA_Models/Laptop/ TDSA_configs/ELMO_Laptop.jsonnet ../amazon/sub_filtered_split_train.txt ../amazon/predicted_targets_train.txt
```