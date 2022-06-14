import streamlit as st
import pandas as pd
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import re, nltk, csv
from PIL import Image
from wordcloud import WordCloud
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


st.set_page_config(layout="wide")

##### HEADER #####

st.title("Natural Language Processing on Twitter")
st.subheader("First Year Project, Project 3: NLP")
st.markdown("""
**IT-University of Copenhagen, Bsc. in Data Science** \\
By Juraj Septak, Gusts Gustav Grīnbergs, Franek Liszka, Mirka Katuscakova and Jannik Elsäßer _(Group E2)_
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
        st.write("Both the hatespeech detection and the emoji predicion dataset came from the same source:")
        st.caption("https://github.com/cardiffnlp/tweeteval")

        st.write("The hatespeech dataset uses the classifiers 1 and 0, hatespeech and not hatespeech, while the emoji dataset had more classifiers:")

        map_df = pd.DataFrame()
        emoji_map = ['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙', '😘', '📷', '🇺🇸', '☀', '💜', '😉', '💯', '😁', '🎄', '📸', '😜']
        map_df["Emoji"] = emoji_map
        map_df = map_df.T
        st.dataframe(map_df)


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

    ############### HATESPEECH Plots
    hsw_stopwords['idx'] = hsw_stopwords.index + 1
    hsw_stopwords['norm_freq'] = hsw_stopwords.frequency / len(hsw_stopwords)
    hsw_stopwords['cumul_frq'] = hsw_stopwords.norm_freq.cumsum()

    sns.set()
    fig, axes = plt.subplots(2,3, figsize=(25,16))
    fig.suptitle("Corpus Frequent Word Statistics", size=30)
    sns.set_theme(style='whitegrid')
    plt.subplots_adjust(hspace = 0.3)

    # axes[0,0].set_xscale('log')
    sns.scatterplot(ax=axes[0,0], x='idx', y='cumul_frq', data=hsw_stopwords).set_title("Hatespeech Cumulative frequency by index", size=18)

    sns.lineplot(x='idx', y='cumul_frq', data=hsw_stopwords[:10000], ax=axes[0,1]).set_title("Hatespeech Cumulative frequency by index, top 10000 tokens", size=18)

    hsw_stopwords['log_frq'] = np.log(hsw_stopwords.frequency)
    hsw_stopwords['log_rank'] = np.log(hsw_stopwords.frequency.rank(ascending=False))
    sns.regplot(x='log_rank', y='log_frq', data=hsw_stopwords, ax=axes[0,2], line_kws={"color": "red"}).set_title("Hatespeech Log-log plot for Zipf's law", size=18)

    ###################### EMOJI PLOTS
    #doing zipfs law on our frq dataframe and plotting
    emoji_stopwords['idx'] = emoji_stopwords.index + 1
    emoji_stopwords['norm_freq'] = emoji_stopwords.frequency / len(emoji_stopwords)
    emoji_stopwords['cumul_frq'] = emoji_stopwords.norm_freq.cumsum()

    # Plots
    # axes[1,0].set_xscale('log')
    sns.scatterplot(ax=axes[1,0], x='idx', y='cumul_frq', data=emoji_stopwords).set_title("Emoji Cumulative frequency by index", size=18)

    sns.lineplot(x='idx', y='cumul_frq', data=emoji_stopwords[:10000], ax=axes[1,1]).set_title("Emoji Cumulative frequency by index, top 10000 tokens", size=18)

    emoji_stopwords['log_frq'] = np.log(emoji_stopwords.frequency)
    emoji_stopwords['log_rank'] = np.log(emoji_stopwords.frequency.rank(ascending=False))
    sns.regplot(x='log_rank', y='log_frq', data=emoji_stopwords, ax=axes[1,2], line_kws={"color": "red"}).set_title("Emoji Log-log plot for Zipf's law", size=18);
    st.pyplot(fig)
    
    return


def man_anot():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("We also did some manual annotation of the hatespeech dataset:")
    st.write("")

    with st.expander("Group manual annotation"):
        st.write("""
        For the manual annotation, a random sample of 100 tweets was selected. 
        To annotate the data semi-automatically, a script was created and run locally. 
        The group members went through the sample independently and without consulting the ground truth,
        and labelled them according to the same scheme. """)

        #Plot (group annotation)
        fig = plt.figure(figsize = (10,2))
        dfcrowd = pd.read_csv("./streamlit/data/survey.csv")
        GT = pd.read_csv("./streamlit/data/GT.csv")
        plt.plot(dfcrowd['ours'],label='Group annotation',color='red',linewidth=3.0)
        plt.plot(GT['value'],label='Original label',color='blue',linewidth=3.0)
        plt.title('Result of our manual annotation compared to the Ground Truth')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0);
        st.pyplot(fig=plt)
    
    with st.expander("Survey"):
        st.write("""In addition to group members annotated the sample,
        survey was created. 11 participants in the age group 18-25,
        assigned if the tweets from the random sample are hate speech or not.""")

        #Plot (survey annotation)
        fig2 = plt.figure(figsize = (10,2))
        plt.plot(dfcrowd['annotation'],label='Crowd annotation',color='red',linewidth=3)
        plt.plot(dfcrowd['ours'],color='green',label='Group annotation',linewidth=3)
        plt.title('Result of our manual annotation compared to the survey')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0);
        #plt.plot(dfcrowd['annotation'],label='Crowd annotation')
        st.pyplot(fig2=plt)
        col1, col2 = st.columns(2)
        with col1:
            fig = plt.figure(figsize = (5,5))
            plt.pie([77,23],labels=['Agreed','Disagreed'],colors=['#eab676','#2596be'],explode=(0, 0.1),autopct='%1.1f%%');
            plt.title('Agreement of survey results and ground truth labels')
            st.pyplot(fig=plt)
        
        with col2: 
            fig = plt.figure(figsize = (5,5))
            plt.pie([81,19],labels=['Agreed','Disagreed'],colors=['#2596be','#eab676'],explode=(0, 0.1),autopct='%1.1f%%');
            plt.title('Agreement of group annotation and ground truth labels')
            st.pyplot(fig=plt)
    
    with st.expander("Inter-annotator Agreement"):
        st.markdown("""
        To report on the inter-annotator we have decided to use the Cohen's kappa as our primary metric, as:
        - Each coder had their own preferences (individual annotator bias)
        - Categories were not equally likely
        As there were more than two annotators, we are using the generalized version 
        of the metric - multi-κ (Fleiss' kappa)""")
        
        #Kappa and agreement table
        d = {'Name': ['Juraj','Mirka','Gust','Jannik','Franek'], 
        'Avg Agreement': [0.73,0.78,0.80,0.70,0.84],
        'Kappa' : [0.4599,0.5600,0.6000,0.3999,0.6799]}
        kappa_df = pd.DataFrame(data=d)
        st.table(kappa_df)

    return 


def auto_predic():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    scores = pd.read_csv("./streamlit/data/hs_scores.csv")
    scores = scores[['F1 score', 'Accuracy Score', 'Recall Score', 'Precision Score']]
    st.table(scores)
    
    lst = scores.values.tolist()
    fig, axes = plt.subplots(figsize=(9, 4))
    x = [1,2,3,4]
    axes.plot(x,multi,label='MultinomialNB', marker='o')
    axes.plot(x,sgd,label='SGD Classifier', marker='o')
    axes.plot(x,knn,label='K-Nearest neighbors', marker='o')
    axes.plot(x,dtc,label='Decision Tree', marker='o')
    axes.plot(x,rf,label='Random Forest', marker='o')
    axes.set_xticks([1,2,3,4])
    axes.set_xticklabels(["F1 Score", "Accuracy Score", "Recall Score", "Precision Score"]), 
    axes.set_title("Hatespeech Different Model Scores")
    axes.legend();

    st.write("""
    Task 4 stuff goes here
    """)

    
    return

def data_aug():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.markdown("## Labeling Trumps twitter insults")
    st.markdown("""
    We took Trumps insults (provided by the New York Times) and combined those with all his other tweets.
    >https://www.nytimes.com/interactive/2021/01/19/upshot/trump-complete-insult-list.html \\
    >https://www.thetrumparchive.com/faq (all Tweets from 2009 to 2020)

    The idea behind this was also that Trump tweets would be very similiar to the data, which our model had been trained on. \\
    The most frequent unique keywords throughout the hatespeech dataset were: 
    > _migrant, refugee, #buildthatwall, bitch, hoe, women_
    These keywords are quite relevant when you look at Donald Trump's presidency, 
    and since all the data was collected during the period of July to September 2018 and, Trump's insult tweet/_normal_ tweet 
    dataset also included tweets from this time period, we hoped to get quite accurate and interesting results.
    """)

    st.write("----------")

    st.write("So using those datasets, and our hatespeech model, we were able to create a dataset with all of Trump's tweets, labelled for being insulting and hatespeech.")

    trump_df = pd.read_csv("./streamlit/data/trump_df.csv")
    trump_df = trump_df[['Labels', 'Tweets', 'HS_Label']]
    trump_df = trump_df.rename(columns={"Insult Labels": "Labels", "Tweets": "Tweets", "HS_Label":"HS_Label"})

    with st.expander("Click here to see what the data frame looks like after labelling each tweet based on our model:"):
        st.table(trump_df.iloc[0:10])

    st.write("Below is a random tweet from our dataset, with it's insult label, and hatespeech probability according to our model:")
    random_tweet = trump_df.iloc[random.randrange(0, len(trump_df), 1)]
    st.markdown(f">_"+random_tweet["Tweets"]+"_")

    st.write("And the same tweet tokenized using our tokenizer:")
    st.code(func_regex(random_tweet["Tweets"]))

    hs_pred, not_hs_pred = classify_and_seperate(str(random_tweet["Tweets"]))

    col1, col2, col3 = st.columns(3)
    col1.metric("Hatespeech Prob.", str(hs_pred*100))
    col2.metric("Not Hatespeech Prob.", str(not_hs_pred*100))
    col3.metric("Insult Label", bool(random_tweet["Labels"]))
    st.write("----------")

    test_input = st.text_input("Input anything here, and see what our model classifies it as:", "Hello there u cunt")

    hs_preda, not_hs_preda = classify_and_seperate(test_input)
    col1a, col2a = st.columns(2)
    col1a.metric("Hatespeech Prob.", str(hs_preda*100))
    col2a.metric("Not Hatespeech Prob.", str(not_hs_preda*100))

    


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

def classify_sentence(text):
    classifier = open_jar('./data/pickle/models/hatespeech_model_MultinomialNB2.sav')
    cv = open_jar('./data/pickle/models/hate/vectorizer.pkl')
    return classifier.predict_proba(cv.transform([text]).toarray())

def open_jar(file_address):
    with open(file_address, 'rb') as f:
        data = pickle.load(f)
    return data

def classify_and_seperate(sentence):
    """Return Hatespeech, Not Hatespeech value"""
    hatespeech_array = classify_sentence(sentence)
    return round(hatespeech_array[0][1], 4), round(hatespeech_array[0][0], 4)

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