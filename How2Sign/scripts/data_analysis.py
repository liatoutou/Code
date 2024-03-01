import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


def get_sentence_descriptive_statistics(data):
    sentence_lengths = data['SENTENCE'].str.split().str.len()  # Number of words in each sentence
    
    total_sen = len(sentence_lengths)
    avg_sent_len = round(np.mean(sentence_lengths),2)
    max_sent_len = np.max(sentence_lengths)
    min_sent_len = np.min(sentence_lengths)


    return total_sen, avg_sent_len, max_sent_len, min_sent_len

def get_word_descriptive_statistics(data):

    # Word Level Analysis
    all_words = [word.lower() for sentence in data['SENTENCE'] for word in sentence.split()]
    all_word_counts = Counter(all_words)

    stop_words = set(stopwords.words('english'))

    filtered_words = [word for word in all_words if word not in stop_words]
    filtered_word_counts = Counter(filtered_words)

    vocab_zize = len(all_word_counts)
    unique_words = len(filtered_word_counts)
    number_of_stop_words = vocab_zize - unique_words
    max_word_len = max([len(word) for word in filtered_words])
    min_word_len = min([len(word) for word in filtered_words])

    avg_word_len = round(np.mean([len(word) for word in filtered_words]), 2)
    

    return vocab_zize, number_of_stop_words, avg_word_len, max_word_len, min_word_len, filtered_words

def print_most_common(data,name):

    all_words = [word.lower() for sentence in data['SENTENCE'] for word in sentence.split()]
    all_word_counts = Counter(all_words)
    stop_words = set(stopwords.words('english'))
    x, y=[], []
    for word,count in all_word_counts.most_common(50):
        if word not in stop_words:
            x.append(word)
            y.append(count)
    sns.barplot(x=y,y=x, palette="rocket")
    plt.title(f'Top 15 words in the {name} set')
    plt.show()

def get_top_ngram(corpus,name, n=None):

    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)[:10]

    x,y=map(list,zip(*words_freq))
    sns.barplot(x=y,y=x, palette="rocket")
    plt.title(f'Top 10 bigrams in the {name} set')
    plt.show()


def plot_statistics(statistics_names, train_stats, val_stats, test_stats):
    
    for i, stat_name in enumerate(statistics_names):
        values = [train_stats[i], val_stats[i], test_stats[i]]
        
        # Create a new figure for each statistic
        plt.figure(figsize=(7, 5))
        plt.bar(['Train', 'Val', 'Test'], values, color=['C0', 'forestgreen', 'firebrick'])
        plt.title(stat_name)
        plt.ylabel('Value')
        
        # Adding the text annotations for each bar
        for j, value in enumerate(values):
            plt.text(j, value, f'{value}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()