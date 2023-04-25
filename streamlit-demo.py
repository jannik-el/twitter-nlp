import streamlit as st
import pandas as pd
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import re, nltk, csv
from PIL import Image
from wordcloud import *
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,classification_report
import sys
import numpy as np
import random
import pickle
import collections
import utils as utl

# st.set_page_config(layout="wide")

##### HEADER #####

st.title("Natural Language Processing on Twitter")
st.subheader("First Year Project, Project 3: NLP")
st.markdown("""
**IT-University of Copenhagen, BSc. in Data Science** \\
By Juraj Septak 🇸🇰, Gusts Gustavs Grīnbergs 🇱🇻, Franek Liszka 🇵🇱, Mirka Katuscakova 🇸🇰 and Jannik Elsäßer 🇮🇪 🇩🇪 _(Group E2)_
""")
st.write("------------------------------------------")
itu_logo = Image.open("./misc/Logo_IT_University_of_Copenhagen.jpg")
st.sidebar.image(itu_logo)


sidebar_options = (
    "Start Page",
    "Tokenizer",
    "Model Demo",
    "Labeling Trump's twitter insults")

##### PAGE CODE ##########

hide_table_row_index = """
        <style>
        tbody th {display:none;}
        .blank {display:none;}
        .row_heading.level0 {display:none;}
        </style>
        """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

def start_page():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("""This interactive app is designed as a representation of our group submission
        for the First Year Project course, using different NLP Machine Learning Models to classify hatespeech, and emojis.
        The hatespeech model is our binary classification task, and the emoji detection is our multi classification task.
        Both datasets consist of data scraped off twitter, and therefore our project is also later in the data augmentation task, focused on twitter datasets.
        On the left hand side you can choose different options from the sidebar.""")
        
    with col2:
    
        im = Image.open("./misc/twitter-logo-2-1.png")
        st.image(im, width=550)

    return

def model_demo():
    st.markdown("## Our Machine Learning Models")

    st.markdown("Below is an interactive example of how our models work:")
    test_input = st.text_input("Input anything here, and see what our model classifies it as:", "Democrats are weak. Hillary to jail. ")

    models = ["SGDC", "DTC", "KNN", "MultinomialNB2", "RF"] 
    emoji_models = ["MultinomialNB", "KNN","SGDC", "DTC"]

    col1f, col2f = st.columns(2)
    hs_mod = col1f.radio("Choose a Hatespeech Model (SGDC is best)", models)
    emo_mod = col2f.radio("Choose an emoji model (MultinomialNB is best)", emoji_models)


    hs_preda, not_hs_preda = classify_and_seperate(test_input, hs_mod)
    hs_preda = str(float(hs_preda)*100)[0:5] 
    not_hs_preda = str(float(not_hs_preda)*100)[0:5]
    emoji_pred = label_to_emoji(test_input, emo_mod)

    col1a, col2a, col3a = st.columns(3)
    col1a.metric("Hatespeech Prob.", f"{hs_preda}%")
    col2a.metric("Not Hatespeech Prob.", f"{not_hs_preda}%")
    col3a.metric("Most likely emoji predicted", emoji_pred)

    # with st.expander("Confusion Matrixes:"):
        # col1, col2 = st.columns(2)
        # with col1:
        #     im = Image.open("./streamlit/data/confusion_matrix_hate.png")
        #     st.image(im, width=750)

        # with col2:
        #     im = Image.open("./streamlit/data/confusion_matrix_emoji.png")
        #     st.image(im, width=750)

    with st.expander("Dataset Labels:"):
        st.write("Both the hatespeech detection and the emoji predicion dataset came from the same source:")
        st.caption("https://github.com/cardiffnlp/tweeteval")

        st.write("The hatespeech dataset uses the classifiers 1 and 0, hatespeech and not hatespeech respectively, while the emoji dataset had more classifiers:")

        map_df = pd.DataFrame()
        emoji_map = ['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙', '😘', '📷', '🇺🇸', '☀', '💜', '😉', '💯', '😁', '🎄', '📸', '😜']
        map_df["Emoji"] = emoji_map
        map_df = map_df.T
        st.dataframe(map_df)

#     with st.expander("Model Scores:"):
#         st.write("(Higher is better)")
#         hate_scores = pd.read_csv("./streamlit/data/hs_scores.csv")
#         hate_scores = hate_scores[['F1 score', 'Accuracy Score', 'Recall Score', 'Precision Score']]
#         hate_scores['Classifier'] = ['DTC', 'K-Nearest neighbors', 'SGDC', 'MultinomialNB', 'Random Forest']
#         hate_scores = hate_scores[["Classifier", "F1 score", "Accuracy Score", "Recall Score", "Precision Score"]]
#         st.table(hate_scores)

#         scores = pd.read_csv("./streamlit/data/emoji_scores.csv")
#         scores = scores[['F1 score', 'Accuracy Score', 'Recall Score', 'Precision Score']]
#         scores['Classifier'] = ['DTC', 'K-Nearest neighbors', 'SGDC', 'MultinomialNB']
#         scores = scores[["Classifier", "F1 score", "Accuracy Score", "Recall Score", "Precision Score"]]
#         st.table(scores)


def tokenizer_page():
    st.markdown("## Tokenizing")


    st.write("We created our own regex tokenizer and looked at how it worked compared to other tokenizers:")

    line = st.text_input('Try it out below:', "Time for some BBQ and whiskey libations. Chomp, belch, chomp! 😍")
    tokenizers = ["Regex", "NLTKTweetModified", "NLTKTweet", "NLTKTreeBank"]

    for i in tokenizers:
        st.write(i, ":")
        st.code(str(tokenize_lines(i, line)))

    return

def trump_demo():
    st.markdown("## Labeling Trump's twitter insults")
    with st.expander("A little background on why we chose this dataset:"):
        st.markdown("""
        We took Trumps insults (provided by the New York Times) and combined those with all his other tweets.
        >https://www.nytimes.com/interactive/2021/01/19/upshot/trump-complete-insult-list.html \\
        >https://www.thetrumparchive.com/faq (all Tweets from 2009 to 2020)

        The idea behind this was also that Trump tweets would be very similiar to the data, which our model had been trained on. \\
        The most frequent unique keywords throughout the hatespeech dataset were: 
        > _migrant, refugee, #buildthatwall, bitch, hoe, women_
        These keywords are quite relevant when you look at Donald Trump's presidency, and since all the data was collected during the \\
        period of July to September 2018 and, Trump's insult tweet/_normal_ tweet dataset also included tweets from this time period, \\
        we hoped to get quite accurate and interesting results.
        """)

    st.write("Using those datasets, and our hatespeech model, we were able to create a dataset with all of Trump's tweets, labelled for being insulting and hatespeech.")

    trump_df = pd.read_csv("./streamlit/data/trump_df.csv")
    trump_df2 = trump_df[['Labels', 'Tweets', 'HS_Label']]
    trump_df2 = trump_df2.rename(columns={"Insult Labels": "Labels", "Tweets": "Tweets", "HS_Label":"HS_Label"})

    st.write("Below is a random tweet from our dataset, with it's insult label, and hatespeech probability according to our model:")
    random_tweet = trump_df.iloc[random.randrange(0, len(trump_df), 1)]
    st.markdown(f">**_"+random_tweet["Tweets"]+"_**")
    st.markdown("_(You can press `r` to refresh and see a new tweet)_")

    st.markdown("""
    Here you can see what label our Model is giving the tweet, and what label was given by the New York Times: \\
    _(Hatespeech ML Model is SGDC, and Emoji Prediction ML Model is KNN)_
    """)

    hs_pred, not_hs_pred = classify_and_seperate(str(random_tweet["Tweets"]))
    hs_pred = str(float(hs_pred)*100)[0:6] 
    not_hs_pred = str(float(not_hs_pred)*100)[0:6]
    trump_emoji = label_to_emoji(str(random_tweet["Tweets"]))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hatespeech Prob.", f"{hs_pred}%")
    col2.metric("Not Hatespeech Prob.", f"{not_hs_pred}%")
    col3.metric("Insult Label (According to NYT)", bool(random_tweet["Labels"]))
    col4.metric("And for fun the emoji prediction:", trump_emoji)    

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

### Classifying based on model:

def classify_sentence(text, model):
    classifier = open_jar(f'./data/pickle/models/hatespeech_model_{model}.sav')
    cv = open_jar('./data/pickle/models/hate/vectorizer.pkl')
    return classifier.predict_proba(cv.transform([text]).toarray())

def open_jar(file_address):
    with open(file_address, 'rb') as f:
        data = pickle.load(f)
    return data

def classify_and_seperate(sentence, model="SGDC"):
    """Return Hatespeech, Not Hatespeech value"""
    hatespeech_array = classify_sentence(sentence, model)
    return '{:.4f}'.format((hatespeech_array[0][1])), '{:.4f}'.format(hatespeech_array[0][0])

def classify_emoji_sentence(text, model="SGDC"):
    classifier = open_jar(f'./data/pickle/models/emoji_{model}.sav')
    if model == "SGDC" or model == "MultinomialNB":
        cv = open_jar('./data/pickle/models/emoji/vectorizer_sss.pkl')
    else:
        cv = open_jar('./data/pickle/models/emoji/vectorizer_normal.pkl')
    data = classifier.predict_proba(cv.transform([text]).toarray())
    return [round(i, 10) for i in data[0]]

def label_to_emoji(text, model="KNN"):
    data = classify_emoji_sentence(text, model)
    data = [float(i) for i in data]
    emoji_map = ['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙', '😘', '📷', '🇺🇸', '☀', '💜', '😉', '💯', '😁', '🎄', '📸', '😜']
    counter=0
    prev=data[0]
    for i, j in enumerate(data):
        if j > prev:
            prev = j
            counter=i
    return emoji_map[counter]

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

def label_predictions(data):
    results = []
    for i in data:
        for j in i:
            if j[1] > j[0]:
                results.append(1)
            else:
                results.append(0)
    return results

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

    mode_two = st.sidebar.radio("Choose a page here:", sidebar_options)
    st.sidebar.success(f"{mode_two} showing on the right:")
    st.sidebar.write("-----------------")

    
    if mode_two == sidebar_options[0]:
        start_page()

    elif mode_two == sidebar_options[1]:
        tokenizer_page()

    elif mode_two == sidebar_options[2]:
        model_demo()

    elif mode_two == sidebar_options[3]:
        trump_demo()


if __name__ == "__main__":
    main()
