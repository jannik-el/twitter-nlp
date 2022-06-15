## This file is for creating functions to shorten down on boilerplate code that we use often
## ie. loading and tokenizing the data

import re
import nltk

################################################################################
#Loading and tokenizing functions

def load_and_tokenize(file_dir, tokenizer="Regex"):
    """
    Takes:
        - file directory of the dataset
        - tokenizer, which tokenizer the data should be tokenized by:
            - NLTKTweetModified
            - NLTKTweet
            - NLTKTreeBank
            - Regex
    Returns:
        - Tokenized dataset as list of lists
    """
    tokenized_data = []

    with open(file_dir, mode ='r')as filer: 
        data = filer.readlines() 
        data = [i.strip("\n") for i in data]

        for line in data:
            if tokenizer == "NLTKTweetModified":
                tokenized_data.append(func_nltktweetmodified(line))

            elif tokenizer == "NLTKTweet":
                tokenized_data.append(func_nltktweet(line))

            elif tokenizer == "NLTKTreeBank":
                tokenized_data.append(func_nltktreebank(line))

            elif tokenizer == "Regex":
                tokenized_data.append(func_regex(line))
    
    return tokenized_data

import pickle

def pickling(data, file_address):
    """
    Takes:
        - data: list of lists, pandas df or etc. that should be pickled
        - file_address: location of where the pickle should be saved
    Returns:
        - None, saves file as pickled file
    """
    with open(file_address, 'wb') as f:
        pickle.dump(data, f)
    return 

def open_jar(file_address):
    """
    Takes:
        - file_address: location of pickle to open
    Returns:
        - data: unpickled data loaded into ram
    """
    with open(file_address, 'rb') as f:
        data = pickle.load(f)
    return data

def func_nltktweetmodified(line):
    filter_list = ['️','', '.', ',', '', '?', '!', '"', '~', "-"]
    tweet_tokenizer = nltk.TweetTokenizer()
    line = [x for x in tweet_tokenizer.tokenize(line) if x not in filter_list]
    return line

def func_nltktweet(line):
    tweet_tokenizer = nltk.TweetTokenizer()
    return tweet_tokenizer.tokenize(line)

def func_nltktreebank(line):
    tree_tokenizer = nltk.TreebankWordTokenizer()
    return tree_tokenizer.tokenize(line)

def func_regex(line):
    filter_list = ['️','', '.', ',', '', '?', '!', '"', '~', "-", "…", ":"]
    token_list = []
    final_list = []
    word = str()

    split_line = re.split(r'\s+|https.*|www.*|http.*|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', line)
    for i in split_line:
        for t in i :
            if t not in filter_list:
                word += t
        if len(word) > 0:
            final_list.append(word)
        else:
            pass
        word=""
    return final_list

###################################################################
# Wordcloud Functions

import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt

def wordcloud(data, stopwords):
    """
    Function for plotting wordcloud in the twitter symbol
    Takes: 
        - data
        - stopwords
    Returns:
        - plt.imshow()
    """
    my_mask = np.array(Image.open('/work/twitter-nlp/misc/Twitter.png'))
    wordcloud = WordCloud(width = 10000, height = 10000, random_state=1, background_color='white',
                        collocations=False, mask=my_mask, contour_width=3, contour_color='#1DA1F2', stopwords=stopwords, min_word_length = 4).generate(data)
    plt.figure(figsize=(20, 20))
    plt.axis("off")
    plt.tight_layout(pad=0)
    return plt.imshow(wordcloud, interpolation='bilinear')

def export_wordcloud(data, stopwords, address):
    """
    Function for plotting wordcloud in the twitter symbol
    Takes: 
        - data
        - stopwords
        - address: directory for saving wordcloud file
    Returns:
        - None, loads image file into address 
    """
    my_mask = np.array(Image.open('/work/twitter-nlp/misc/Twitter.png'))
    wordcloud = WordCloud(width = 10000, height = 10000, random_state=1, background_color='white',
                        collocations=False, mask=my_mask, contour_width=3, contour_color='#1DA1F2', stopwords=stopwords, min_word_length = 4).generate(data.lower())
    plt.figure(figsize=(20, 20))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')
    with open(f"{address}/Output.svg", "w") as text_file:
        text_file.write(wordcloud.to_svg())
    wordcloud.to_file("wordcloud1.png")