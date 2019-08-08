python word_embeddings/create_embeddings.py ../yelp/splits/filtered_split_train.txt 1 ../yelp/embeddings/yelp_300
python word_embeddings/create_embeddings.py ../yelp/splits/filtered_split_train.txt 3 ../yelp/embeddings/yelp_300_phrases_3
python word_embeddings/create_embeddings.py ../amazon/filtered_split_train.txt 1 ../amazon/embeddings/amazon_300
python word_embeddings/create_embeddings.py ../amazon/filtered_split_train.txt 3 ../amazon/embeddings/amazon_300_phrases_3
python word_embeddings/create_embeddings.py ../MP-Tweets/filtered_split_train.txt 1 ../MP-Tweets/embeddings/mp_300
python word_embeddings/create_embeddings.py ../MP-Tweets/filtered_split_train.txt 3 ../MP-Tweets/embeddings/mp_300_phrases_3