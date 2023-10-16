import sys
from collections import defaultdict, Counter
import math
import random
import os
import os.path

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  


def get_ngrams(sequence, n):

    #0. error check
    if n <= 0:
        raise ValueError("negative number for ngram, won't work")

    #1. add padding to beginning and end of input string
    padded_strings = ["START"] * (n - 1) + sequence + ["END"] 

    #2. generate n-grams; store them in a list of tuples
    ngrams = []
    for i in range(len(padded_strings) - n + 1):
        ngram = tuple(padded_strings[i:i + n])
        ngrams.append(ngram)

    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        #1. Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.sentence_count = 0
    
        #2. Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        #3. iterate through the corpus one more time to get total word count 
        generator = corpus_reader(corpusfile) #not lexical, raw word count 
        self.totalwords = sum(len(sentence) for sentence in generator)       
          

    def count_ngrams(self, corpus):
  
        #0. open up the dictionaries to fill later
        self.unigramcounts = Counter()
        self.bigramcounts = Counter()
        self.trigramcounts = Counter() 

        #1. iterate over every sentence in the generator
        for sentence in corpus:
          unigrams = get_ngrams(sentence, 1) 
          bigrams = get_ngrams(sentence, 2)
          trigrams = get_ngrams(sentence, 3)

          #2. add this sentence's grams to the grand gram dictionaries 
          self.unigramcounts.update(unigrams) 
          self.bigramcounts.update(bigrams)         
          self.trigramcounts.update(trigrams)

          self.sentence_count += 1
  
        return self.unigramcounts, self.bigramcounts, self.trigramcounts

      
    def raw_trigram_probability(self,trigram):
      
        numerator = self.trigramcounts[trigram]
 
        bigram = (trigram[0], trigram[1])   
        if bigram == (('START', 'START')):
          return numerator / self.sentence_count

        denominator = self.bigramcounts[bigram]
        if denominator == 0:
          return 1/(len(self.lexicon)+1)
        else: 
          return numerator / float(denominator)

  
    def raw_bigram_probability(self, bigram):
      
        numerator = self.bigramcounts[bigram]

        unigram = (bigram[0],)
        if unigram == (('START',)):
          return numerator / self.sentence_count
          
        denominator = self.unigramcounts[unigram] #use it to find it's frequency in the unigram dict
        if denominator == 0:
          return 1/(len(self.lexicon)+1)
        else: 
          return numerator / float(denominator)
          
  
    def raw_unigram_probability(self, unigram):

      numerator = self.unigramcounts[unigram]
      denominator = self.totalwords

      if numerator == 0:
        return 1/self.totalwords

      return numerator / float(denominator)

    def smoothed_trigram_probability(self, trigram):

      lambda1 = 1/3.0
      lambda2 = 1/3.0
      lambda3 = 1/3.0

      bigram = (trigram[1], trigram[2])
      unigram = (trigram[2],)
      
      a = lambda1 * self.raw_unigram_probability(unigram)
      b = lambda2 * self.raw_bigram_probability(bigram)
      c = lambda3 * self.raw_trigram_probability(trigram)
               
      return a + b + c
  
    def sentence_logprob(self, sentence):

      prob = 0 
      trigrams = get_ngrams(sentence, 3)
      
      for trigram in trigrams:
        smoothed_prob = self.smoothed_trigram_probability(trigram)
        prob += math.log2(smoothed_prob)
        
      return float(prob)

    def perplexity(self, corpus):

      totalwords = 0
      sumlog = 0

      for sentence in corpus: 
        sumlog += self.sentence_logprob(sentence)
        totalwords += len(sentence)

      return 2**(-sumlog/totalwords)
      

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
          
            if pp1 < pp2:
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
    
            if pp2 < pp1:
                correct += 1
            total += 1 
            
            return correct/total

if __name__ == "__main__":
 
    acc = essay_scoring_experiment("train_high.txt", "train_low.txt", "test_high", "test_low")
    print(acc)

