# Trigram-Language-Model
Essay classification script using a trigram language model. 

This project implements a trigram language model in Python, designed to process, analyze, and classify text based on a reference corpus. The foundational feature of this model is the extraction of n-grams from sentences, computing smoothed probabilities for each trigram, and applying the model to a collection of essays to be classified according to their command of English. 

As the code traverses through a corpus, it counts the occurrences of n-grams using Python dictionaries to generate raw probabilities for each. Using these raw probability calculations, the model computes smoothed probabilities for each trigram, bigram, and unigram. Linear interpolation is used to offer a balanced probability estimation for each n-gram to make calculating the probability of an entire sentence or sequence possible and accurate. 

The perplexity metric included in the script is used to check how well the probability distribution predicted by the model aligns with the actual distribution of the words in the corpus.

The application of this model is to classify essays written by ESL speakers as either 'high' or 'low' in their command of English. By training the trigram model on both high and low-quality essays and then evaluating unseen essays based on the perplexity scores derived from these models, the system can predict the quality of essays with an accuracy of 84%.
