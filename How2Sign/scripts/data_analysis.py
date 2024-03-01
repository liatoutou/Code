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



#load data
df_train = pd.read_csv('data/how2Sign_train.csv', sep='\t')
df_val = pd.read_csv('data/how2Sign_val.csv', sep='\t')
df_test = pd.read_csv('data/how2Sign_test.csv', sep='\t')




def get_descriptive_statistics(data):
    sentence_lengths = data['SENTENCE'].str.split().str.len()  # Number of words in each sentence
    
    total_sen = len(sentence_lengths)
    avg_sent_len = round(np.mean(sentence_lengths))
    max_sent_len = np.max(sentence_lengths)
    min_sent_len = np.min(sentence_lengths)

    # Word Level Analysis
    all_words = [word.lower() for sentence in data['SENTENCE'] for word in sentence.split()]
    all_word_counts = Counter(all_words)
    # Get English stop words
    stop_words = set(stopwords.words('english'))

    # Exclude stop words from the word list
    filtered_words = [word for word in all_words if word not in stop_words]
    filtered_word_counts = Counter(filtered_words)

    # Recalculate word level statistics without stop words
    total_words = len(all_word_counts)
    # Get the number of unique words
    unique_words = len(filtered_word_counts)
    number_of_stop_words = total_words - unique_words
    #round to 2 decimal places
    avg_word_len = round(np.mean([len(word) for word in filtered_words]), 2)


    filtered_word_counts1 = filtered_word_counts.most_common(15)


    return total_sen, avg_sent_len, max_sent_len, min_sent_len, total_words, unique_words, number_of_stop_words, avg_word_len, filtered_word_counts1, filtered_word_counts

# Get descriptive statistics for each dataset
train_total_sen, train_mean_sentence_length, train_max_sentence_length, train_min_sentence_length, train_total_words, unique_words_train, number_of_stop_words_train, avg_word_len_train, most_common_words_train,filtered_words_train = get_descriptive_statistics(df_train)
val_total_sen, val_mean_sentence_length, val_max_sentence_length, val_min_sentence_length, val_total_words, unique_words_val, number_of_stop_words_val, avg_word_len_val, most_common_words_val, filtered_words_val = get_descriptive_statistics(df_val)
test_total_sen, test_mean_sentence_length, test_max_sentence_length, test_min_sentence_length, test_total_words, unique_words_test, number_of_stop_words_test, avg_word_len_test, most_common_words_test, filtered_words_test= get_descriptive_statistics(df_test)

train_stats = [train_total_sen, train_mean_sentence_length, train_max_sentence_length, train_min_sentence_length, train_total_words, unique_words_train, number_of_stop_words_train, avg_word_len_train]
val_stats = [val_total_sen, val_mean_sentence_length, val_max_sentence_length, val_min_sentence_length, val_total_words, unique_words_val, number_of_stop_words_val, avg_word_len_val]
test_stats = [test_total_sen, test_mean_sentence_length, test_max_sentence_length, test_min_sentence_length, test_total_words, unique_words_test, number_of_stop_words_test, avg_word_len_test]

statistics_names = [
    'Total Sentences', 'Average Sentence Length', 'Maximum Sentence Length',
    'Minimum Sentence Length', 'Total Words', 'Unique Words', 'Number of Stop Words','Average Word Length', 
]

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

def print_most_common(most_common,name):
    x, y=[], []
    for word,count in most_common:
            x.append(word)
            y.append(count)
    sns.barplot(x=y,y=x, palette="rocket")
    plt.title(f'Top 15 words in the {name} set')
    plt.show()

def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]

print_most_common(most_common_words_train, "training")
print_most_common(most_common_words_val, "validation")
print_most_common(most_common_words_test, "test")

# Get the top 10 bigrams and trigrams
train_bigrams = get_top_ngram(filtered_words_train, 2)[:10]
val_bigrams = get_top_ngram(filtered_words_val, 2)[:10]
test_bigrams = get_top_ngram(filtered_words_test, 2)[:10]

x,y=map(list,zip(*train_bigrams))
sns.barplot(x=y,y=x, palette="rocket")
plt.title('Top 10 bigrams in the training set')
plt.show()


x,y=map(list,zip(*val_bigrams))
sns.barplot(x=y,y=x, palette="rocket")
plt.title('Top 10 bigrams in the validation set')
plt.show()

x,y=map(list,zip(*test_bigrams))
sns.barplot(x=y,y=x, palette="rocket")
plt.title('Top 10 bigrams in the test set')
plt.show()