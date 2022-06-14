import streamlit as st
import pandas as pd
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import re, nltk, csv
from PIL import Image

st.set_page_config(layout="wide")

##### HEADER #####

st.title("Natural Language Processing")
st.subheader("First Year Project, Twitter NLP")
st.caption("*IT-University of Copenhagen, Bsc. in Data Science*")
st.caption("By Juraj Septak, Gusts Gustav, Franek Liszka, Mirka and Jannik Elsäßer *(Group E2)*")
st.write("------------------------------------------")

sidebar_options = (
    "Start Page", 
    "Preprocessing",
    "Data Characterisation", 
    "Manual Annotation", 
    "Automatic Prediction", 
    "Data Augmentation")

melanoma_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Twitter-logo.svg/768px-Twitter-logo.svg.png"
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

        with open('./data/tweeteval/datasets/emoji/mapping.txt', mode='r') as mapping:
            map_data = mapping.readlines()
            map_data = [i.strip("\n") for i in map_data]
            # st.write(map_data)     


    with st.expander("Testing Tokenizers"):
        st.write("We created our own regex tokenizer and looked at how it worked compared to other tokenizers:")

        line = st.text_input('Try it out below:', "#Fabulous evening tonight, I'm so @home 😍")
        tokenizers = ["Regex", "NLTKTweetModified", "NLTKTweet", "NLTKTreeBank"]

        for i in tokenizers:
            st.write(i, ":")
            st.code(str(tokenize_lines(i, line)))

    with st.expander("Comparing Tokenizers on the hatespeech dataset"):
        st.write("Dont Break")
        

def data_char():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("some text comes here")

    corpus_data = pd.read_csv("./streamlit/data/corpus_data.csv")
    corpus_data = corpus_data[['Dataset', 'Corpus size', 'Vocabulary size', 'Type to Token ratio']]
    st.dataframe(corpus_data)

    hsw_stopwords = pd.read_csv("./streamlit/data/hsw_stopwords.csv")
    #hsw_stopwords.iloc[0:50].plot.bar(x='token',y='frequency', figsize=(17,5))
    st.pyplot(hsw_stopwords.iloc[0:50])
    #plt.title("Most frequent words in hatespeech dataset (top 50) without stopwords")
    #plt.xticks(rotation = 90)


    return


def take_pic_page():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    take_picture = st.camera_input("Take a picture to test it for melanoma:")

    if take_picture:
        st.image(take_picture)
    
    st.write("Currently it just shows the picture, but here we can build the algorithm in and immediately show the results which would be really cool")
    return

def test_bulk_img():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("""
    The idea is that they can upload a folder with a bunch of images and test them on this page.
    This might be a bit difficult to implement. 
    """)
    return

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
        take_pic_page()

    elif app_mode == sidebar_options[4]:
        test_bulk_img()

if __name__ == "__main__":
    main()