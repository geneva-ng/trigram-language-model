# Trigram-Language-Model
Essay classification script using a trigram language model. 

This project implements a trigram language model in Python, designed to process, analyze, and generate text based on given specifications. The foundational feature of this model is the extraction of n-grams from sentences. 

As the code traverses through a corpus, it counts the occurrences of n-grams using Python dictionaries. Based on these counts, raw probabilities for each type of n-gram are calculated. Using these raw probability calculations, the model computes smoothed probabilities for each trigram, bigram, and unigram. Linear interpolation is used to offer a balanced probability estimation for each n-gram to make calculating the probability of an entire sentence or sequence possible. 

The perplexity metric included in the script is used to check how well the probability distribution predicted by the model aligns with the actual distribution of the words in the corpus.

The application of this model is to classify essays written by ESL speakers as either 'high' or 'low' in their command of English. By training separate trigram models on both high and low-quality essays and then evaluating unseen essays based on the perplexity scores derived from these models, the system can predict the quality of essays with an accuracy of 84%.
