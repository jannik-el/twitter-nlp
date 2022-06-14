import streamlit as st
import pandas as pd
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import re, nltk, csv
from PIL import Image
from wordcloud import WordCloud

st.set_page_config(layout="wide")

##### HEADER #####

st.title("Natural Language Processing on Twitter")
st.subheader("First Year Project, Project 3: NLP")
st.markdown("""
**IT-University of Copenhagen, Bsc. in Data Science** \\
By Juraj Septak, Gusts Gustav, Franek Liszka, Mirka and Jannik Elsäßer _(Group E2)_
""")
st.write("------------------------------------------")

sidebar_options = (
    "Start Page", 
    "Preprocessing",
    "Data Characterisation", 
    "Manual Annotation", 
    "Automatic Prediction", 
    "Data Augmentation")

##### PAGE CODE ##########

def start_page():
    st.sidebar.write("---------------------")
    st.sidebar.success("Start Page showing on the right:")
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        This interactive app is designed as a representation of our groups submission
        for the First Year Project 2, using different NLP Machine Learning Models to classify hatespeech, and emojis.
        The hatespeech model is our binary classification task, and the emoji detection is our multi classification task.
        Both datasets consist of data scraped off twitter, and therefore our project is also later in the data augmentation task, focussed on twitter datasets.
        On the left hand side you can choose different options from the sidebar.
        These are all different tasks of our project.  
        """)


    with col2:
        im = Image.open("./misc/Twitter.png")
        st.image(im, caption='Put Twitter Word Cloud image Here', width=400)

    return

def preprocessing():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("To be able to create a model for the different tasks we had, we first had to do some prepocessing.")

    with st.expander("Our Datasets"):
        st.write("Both the hatespeech-, and the emoji-detection dataset came from the same source:")
        st.caption("https://github.com/cardiffnlp/tweeteval")

        st.write("The hatespeech dataset uses the classifiers 1 and 0, hatespeech and not hatespeech, while the emoji dataset had more classifiers:")

        map_df = pd.DataFrame()
        emoji_map = ['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙', '😘', '📷', '🇺🇸', '☀', '💜', '😉', '💯', '😁', '🎄', '📸', '😜']
        map_df["Emoji"] = emoji_map
        map_df = map_df.T
        st.table(map_df)


    with st.expander("Testing Tokenizers"):
        st.write("We created our own regex tokenizer and looked at how it worked compared to other tokenizers:")

        line = st.text_input('Try it out below:', "Time for some BBQ and whiskey libations. Chomp, belch, chomp! (@ Lucille's Smokehouse Bar-B-Que)😍")
        tokenizers = ["Regex", "NLTKTweetModified", "NLTKTweet", "NLTKTreeBank"]

        for i in tokenizers:
            st.write(i, ":")
            st.code(str(tokenize_lines(i, line)))

    with st.expander("Comparing Tokenizers on the hatespeech dataset"):
        st.write("First we looked at how many tokens are 'leftover', after tokenizing:")
    
        token_stats = pd.DataFrame()
        token_stats["Regex"] = [185976]
        token_stats["NLTKTweetModified"] = [197307]
        token_stats["NLTKTweet"] = [213182]
        token_stats["NLTKTreeBank"] = [218934]
        # token_stats = token_stats.transpose()
        # token_stats.columns.name = 'No. Tokens'
        st.table(token_stats)
    
        st.write("We also looked at how the tokenizers compare across the top 100 most frequent tokens:")

        col1, col2, col3 = st.columns(3)

        with col2:
            st.image("./streamlit/data/token_comparison.png", width = 500)
            st.write("TODO, change this image to svg cus its shite quality")
    return
    
def data_char():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    corpus_data = pd.read_csv("./streamlit/data/corpus_data.csv")
    corpus_data = corpus_data[['Dataset', 'Corpus size', 'Vocabulary size', 'Type to Token ratio']]
    st.dataframe(corpus_data)

    hsw_stopwords = pd.read_csv("./streamlit/data/hsw_stopwords.csv")
    emoji_stopwords = pd.read_csv("./streamlit/data/emojiw_stopwords.csv")
    # creating dicts, to make wordclouds
    hs_wo_stopwords_dict = dict(zip(list(hsw_stopwords['token']), list(hsw_stopwords['frequency'])))
    emoji_wo_stopwords_dict = dict(zip(list(emoji_stopwords['token']), list(emoji_stopwords['frequency'])))
    hs_wo_stopwords_dict_lf = dict((k, v) for k, v in hs_wo_stopwords_dict.items() if v <= 3) #least frequent hate (value <= 3)
    emoji_wo_stopwords_dict_lf = dict((k,v) for k, v in emoji_wo_stopwords_dict.items() if v <= 3) #least frequent emoji (value <=3)

    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure(figsize = (5,10))
        hsw_stopwords.iloc[0:30].plot.barh(x='token',y='frequency', figsize=(5,14))
        plt.title("Most frequent words in hatespeech dataset (top 50) without stopwords")
        plt.xticks(rotation = 90)
        st.pyplot(fig=plt)
        st.write("**Least frequent words in hatespeech dataset**")
        custom_wc(hs_wo_stopwords_dict_lf)
        
    with col2: 
        fig = plt.figure(figsize = (5,10))
        emoji_stopwords.iloc[0:30].plot.barh(x='token',y='frequency', figsize=(5,13))
        plt.title("Most frequent words in emoji dataset (top 50) without stopwords")
        plt.xticks(rotation = 90)
        st.pyplot(fig=plt)
        st.write("**Least frequent words in emoji dataset**")
        custom_wc(emoji_wo_stopwords_dict_lf)

    return


def man_anot():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("We also did some manual annotation of the hatespeech dataset:")
    col1, col2 = st.columns(2)
    st.write("Place noice graphics and shit here, can do it with columns, ask jannik")

    return

def auto_predic():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("""
    Task 4 stuff goes here
    """)
    return

def data_aug():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("""
    Task 5 shit goes here
    """)

############## NLP Code ###################

#### Tokenizers:

def tokenize_lines(tokenizer, line):
    if tokenizer == "NLTKTweetModified":
        return func_nltktweetmodified(line)

    elif tokenizer == "NLTKTweet":
        return func_nltktweet(line)

    elif tokenizer == "NLTKTreeBank":
        return func_nltktreebank(line)

    elif tokenizer == "Regex":
        return func_regex(line)

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

def custom_wc(data):
    wordcloud = WordCloud(background_color = 'white',
                        width = 1200,
                        height = 1000)
    wordcloud.generate_from_frequencies(data)
    plt.figure(figsize=(5,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot(fig=plt)


###### DOWNLOADING IMAGE DATA CODE ###############

image_url = "https://github.com/jannik-el/melanoma-detection-app/blob/main/data/example-images/ISIC_0001769.jpg?raw=true"
mask_url = "https://github.com/jannik-el/melanoma-detection-app/blob/main/data/example-mask/ISIC_0001769_segmentation.png?raw=true"

def download_image(url, name):
    download = requests.get(url).content
    f = open(f'{name}.jpg', 'wb')
    f.write(download)
    f.close()
    return f"{name}.jpg"

######## OTHER BOILERPLATE CODE ##############

def plot_image(image):
    fig, ax = plt.subplots()
    ima=np.array(Image.open(image))
    ax.imshow(ima)
    return st.pyplot(fig)

###### MAIN FUNCTION #################

def main():

    st.sidebar.title("Explore the following:")
    st.sidebar.write("---------------------")
    app_mode = st.sidebar.selectbox("Please select from the following:", sidebar_options)

    if app_mode == "Start Page":
        start_page()

    elif app_mode == sidebar_options[1]:
        preprocessing()

    elif app_mode == sidebar_options[2]:
        data_char()

    elif app_mode == sidebar_options[3]:
        man_anot()

    elif app_mode == sidebar_options[4]:
        auto_predic()

    elif app_mode == sidebar_options[5]:
        data_aug()

if __name__ == "__main__":
    main()