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
import collections
import utils as utl
from sklearn.metrics import plot_confusion_matrix

st.set_page_config(layout="wide")

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
    "Preprocessing",
    "Data Characterisation", 
    "Manual Annotation", 
    "Automatic Prediction", 
    "Data Augmentation")

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


    
    col1, col2 = st.columns(2)

    with col1:
        st.write("""**This interactive app is designed as a representation of our group submission
        for the First Year Project course, using different NLP Machine Learning Models to classify hatespeech, and emojis.
        The hatespeech model is our binary classification task, and the emoji detection is our multi classification task.
        Both datasets consist of data scraped off twitter, and therefore our project is also later in the data augmentation task, focused on twitter datasets.
        On the left hand side you can choose different options from the sidebar.
        These are all different tasks of our project.**""")
        
    with col2:
    
        im = Image.open("./misc/wordart.png")
        st.image(im, width=550)

    return

def preprocessing():

    st.write("To be able to create a model for the different tasks we had, we first had to do some prepocessing.")

    with st.expander("Our datasets"):
        st.write("Both the hatespeech detection and the emoji predicion dataset came from the same source:")
        st.caption("https://github.com/cardiffnlp/tweeteval")

        st.write("The hatespeech dataset uses the classifiers 1 and 0, hatespeech and not hatespeech respectively, while the emoji dataset had more classifiers:")

        map_df = pd.DataFrame()
        emoji_map = ['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙', '😘', '📷', '🇺🇸', '☀', '💜', '😉', '💯', '😁', '🎄', '📸', '😜']
        map_df["Emoji"] = emoji_map
        map_df = map_df.T
        st.dataframe(map_df)


    with st.expander("Testing tokenizers"):
        st.write("We created our own regex tokenizer and looked at how it worked compared to other tokenizers:")

        line = st.text_input('Try it out below:', "Time for some BBQ and whiskey libations. Chomp, belch, chomp! (@ Lucille's Smokehouse Bar-B-Que)😍")
        tokenizers = ["Regex", "NLTKTweetModified", "NLTKTweet", "NLTKTreeBank"]

        for i in tokenizers:
            st.write(i, ":")
            st.code(str(tokenize_lines(i, line)))

    with st.expander("Comparing tokenizers on the hatespeech dataset"):
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

    with st.expander("Corpora statistics"):
        corpus_data = pd.read_csv("./streamlit/data/corpus_data.csv")
        corpus_data = corpus_data[['Dataset', 'Corpus size', 'Vocabulary size', 'Type to Token ratio']]
        st.dataframe(corpus_data)

    with st.expander("Most frequent tokens"):

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

    st.write("")

    with st.expander("Group manual annotation"):
        st.write("""
        For the manual annotation, a random sample of 100 tweets from the hate speech dataset was selected. 
        To annotate the data semi-automatically, a script was created and run locally. 
        The group members went through the sample independently and without consulting the ground truth,
        and labelled them according to the same scheme. """)

        st.write("")

        st.markdown("""
        For disagreement in 19 cases the reasons could be:
        - the tweets were manually labelled before reading the definition of hate speech (some group members could already have prior knowledge)
        - some tweets could be understood in varying ways depending on the context
        - for some tweets, the intentions of the author were unclear (given tweet could be understood as a joke in some settings)
        """)
        
        st.write("")

        #Plot (group annotation)
        fig = plt.figure(figsize = (10,2))
        dfcrowd = pd.read_csv("./streamlit/data/survey.csv")
        GT = pd.read_csv("./streamlit/data/GT.csv")
        plt.plot(dfcrowd['ours'],label='Group annotation',color='red',linewidth=3.0)
        plt.plot(GT['value'],label='Original label',color='blue',linewidth=3.0)
        plt.title('Result of our manual annotation compared to the Ground Truth')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0);
        st.pyplot(fig=plt)
    
    with st.expander("Survey (external annotation)"):
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

    with st.expander("Agreement with the ground truth"):
        st.write("""We have looked at the agreeement with the ground truth provided
         along with the dataset. It was annotated by annotators (crowd)
          and 2 experts (native or near-native speakers of British English, 
          having an extensive experience in annotating data for the task's subject).""")

        col1, col2 = st.columns(2)
        with col1:
            fig = plt.figure(figsize = (5,5))
            plt.pie([81,19],labels=['Agreed','Disagreed'],colors=['#2596be','#eab676'],explode=(0, 0.1),autopct='%1.1f%%');
            plt.title('Agreement of group annotation and ground truth labels')
            st.pyplot(fig=plt)
        
        with col2: 
            fig = plt.figure(figsize = (5,5))
            plt.pie([77,23],labels=['Agreed','Disagreed'],colors=['#eab676','#2596be'],explode=(0, 0.1),autopct='%1.1f%%');
            plt.title('Agreement of survey results and ground truth labels')
            st.pyplot(fig=plt)
    with st.expander("Tweets we have not agreed on"):
        tweets=pd.read_csv("./streamlit/data/tweets.csv")
        dfpd=pd.read_csv("./streamlit/data/dfpd.csv")
        dfpd=dfpd[['Ground Truth','Tweet']]
        st.table(dfpd)
        
    with st.expander("Inter-annotator agreement"):
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
    st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = plt.figure(figsize = (2,2))
        y_train_counts = open_jar("./streamlit/ytraincounts.pkl")
        plt.rcParams['font.size'] = 16.0
        plt.pie(collections.Counter(list(y_train_counts)).values(), labels=['Not hate speech','Hate speech'],colors=['#00695c','#b71c1c'],explode=(0, 0.1), autopct = lambda p:f'{p:.2f}%', textprops={'fontsize': 2})
        st.pyplot(fig1=plt)
    with col2:
        fig2 = plt.figure(figsize = (30,30))
        y_train_emoji_counts = open_jar("./streamlit/ytrainemojicounts.pkl")
        emoji_classes= pd.read_csv("./streamlit/data/mapping-2.txt", sep = "	", header=None)
        plt.rcParams['font.size'] = 11.0
        plt.pie(collections.Counter(list(y_train_emoji_counts)).values(), labels=list(emoji_classes[2]), autopct = lambda p:f'{p:.2f}%', textprops={'fontsize': 22});
        st.pyplot(fig2=plt)

    st.write("------------------------------------------------------------------")
    
    hate_scores = pd.read_csv("./streamlit/data/hate_scores.csv")
    hate_scores = hate_scores[['F1 score', 'Accuracy Score', 'Recall Score', 'Precision Score']]
    
    lst = hate_scores.values.tolist()
    fig, axes = plt.subplots(figsize=(10, 4))
    x = [1,2,3,4]
    axes.plot(x,lst[0],label='DTC', marker='o')
    axes.plot(x,lst[1],label='K-Nearest neighbors', marker='o')
    axes.plot(x,lst[2],label='SGDC', marker='o')
    axes.plot(x,lst[3],label='MultinomialNB', marker='o')
    axes.plot(x,lst[4],label='Random Forest', marker='o')
    axes.set_xticks([1,2,3,4])
    axes.legend()
    axes.set_xticklabels(["F1 score", "Accuracy Score", "Recall Score", "Precision Score"])
    axes.set_title("Hatespeech Different Model Scores")
    st.pyplot()

    hate_scores['Classifier'] = ['DTC', 'K-Nearest neighbors', 'SGDC', 'MultinomialNB', 'Random Forest']
    hate_scores = hate_scores[["Classifier", "F1 score", "Accuracy Score", "Recall Score", "Precision Score"]]
    st.table(hate_scores)

    # im = Image.open("./streamlit/data/confusion_matrix_emoji.png")
    # st.image(im, width=700)
    _, ax = plt.subplots(figsize=(20,20))
    classifier = open_jar("./streamlit/data/ML_Confusion_matrix/classifier_emoji.pkl")
    X_test = open_jar("./streamlit/data/ML_Confusion_matrix/X_test_emoji.pkl")
    y_test = open_jar("./streamlit/data/ML_Confusion_matrix/y_test_emoji.pkl")
    plot_confusion_matrix(classifier, X_test, y_test, ax = ax)
    st.pyplot()

    st.write("------------------------------------------------------------------")

    scores = pd.read_csv("./streamlit/data/emoji_scores.csv")
    scores = scores[['F1 score', 'Accuracy Score', 'Recall Score', 'Precision Score']]
    
    lst = scores.values.tolist()
    fig, axes = plt.subplots(figsize=(10, 4))
    x = [1,2,3,4]
    axes.plot(x,lst[0],label='DTC', marker='o')
    axes.plot(x,lst[1],label='K-Nearest neighbors', marker='o')
    axes.plot(x,lst[2],label='SGDC', marker='o')
    axes.plot(x,lst[3],label='MultinomialNB', marker='o')
    axes.set_xticks([1,2,3,4])
    axes.legend()
    axes.set_xticklabels(["F1 score", "Accuracy Score", "Recall Score", "Precision Score"])
    axes.set_title("Emoji Different Model Scores")
    st.pyplot()

    scores['Classifier'] = ['DTC', 'K-Nearest neighbors', 'SGDC', 'MultinomialNB']
    scores = scores[["Classifier", "F1 score", "Accuracy Score", "Recall Score", "Precision Score"]]
    st.table(scores)

    st.write("------------------------------------------------------------------")

    st.markdown("Below is an interactive example of how our models work:")
    test_input = st.text_input("Input anything here, and see what our model classifies it as:", "Democrats Hillary Weak #MAGA")

    models = ["SGDC", "DTC", "KNN", "MultinomialNB2", "RF"]
    emoji_models = ["KNN","SGDC", "DTC", "MultinomialNB"]

    col1f, col2f = st.columns(2)
    hs_mod = col1f.radio("Choose a Hatespeech Model (SGDC is best)", models)
    emo_mod = col2f.radio("Choose an emoji model (KNN is best)", emoji_models)


    hs_preda, not_hs_preda = classify_and_seperate(test_input, hs_mod)
    hs_preda = str(float(hs_preda)*100)[0:5] 
    not_hs_preda = str(float(not_hs_preda)*100)[0:5]
    emoji_pred = label_to_emoji(test_input, emo_mod)

    col1a, col2a, col3a = st.columns(3)
    col1a.metric("Hatespeech Prob.", f"{hs_preda}%")
    col2a.metric("Not Hatespeech Prob.", f"{not_hs_preda}%")
    col3a.metric("Most likely emoji predicted", emoji_pred)
    
    
    
    return

def data_aug():

    st.write("Using our models, we decided to looked at two datasets that we thought could prove interesting results:")
    st.markdown("## 1. Labeling Trump's twitter insults")
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

    st.write("So using those datasets, and our hatespeech model, we were able to create a dataset with all of Trump's tweets, labelled for being insulting and hatespeech.")

    trump_df = pd.read_csv("./streamlit/data/trump_df.csv")
    trump_df2 = trump_df[['Labels', 'Tweets', 'HS_Label']]
    trump_df2 = trump_df2.rename(columns={"Insult Labels": "Labels", "Tweets": "Tweets", "HS_Label":"HS_Label"})

    with st.expander("Click here to see what the data frame looks like after labelling each tweet based on our model:"):
        st.table(trump_df2.iloc[0:10])

    st.write("Below is a random tweet from our dataset, with it's insult label, and hatespeech probability according to our model:")
    random_tweet = trump_df.iloc[random.randrange(0, len(trump_df), 1)]
    st.markdown(f">_"+random_tweet["Tweets"]+"_")
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
    col3.metric("Insult Label", bool(random_tweet["Labels"]))
    col4.metric("And for fun the emoji prediction:", trump_emoji)
    
    st.markdown("### The results across the entire dataset:")

    #### Trump DF results code ###

    with st.expander("Click here to see the results:"):

        combined_df = trump_df
        data = [len(combined_df[combined_df["Labels"] == 0])/len(combined_df),len(combined_df[combined_df["Labels"] == 1])/len(combined_df)]
        data2 = [len(combined_df[combined_df["HS_Label"] == 0])/len(combined_df), len(combined_df[combined_df["HS_Label"] == 1])/len(combined_df)]
        labels = ['Insult', 'Not Insult']
        labels2 = ['Hatespeech', "Not Hatespeech"]
        plt.rcParams['font.size'] = 8.0
        fig1, ax = plt.subplots(1,2, figsize=(12,5))
        _,_,autotexts0=ax[0].pie(data, labels=labels,colors=['#00695c','#b71c1c'],explode=(0, 0.1), autopct='%1.1f%%', shadow=True)
        _,_,autotexts1=ax[1].pie(data2, labels=labels2, autopct='%1.1f%%',colors=['#00695c','#b71c1c'],explode=(0, 0.1), shadow=True)
        for autotext in autotexts0:
            autotext.set_color('white')
        for autotext in autotexts1:
            autotext.set_color('white')
        plt.tight_layout()
        fig1.suptitle("Insult label ratio to hatespeech label ratio comparison:")
        st.pyplot(fig1)

    st.markdown("### Looking more closely however:")

    with st.expander("Click here to see the results:"):

        combined_df = pd.read_csv("./streamlit/data/trump_agreement.csv")

        agreement_ratio = [
        len(combined_df[combined_df["Agreement"] == "Agree Hatespeech"])/len(combined_df),
        len(combined_df[combined_df["Agreement"] == "Agree Not Hatespeech"])/len(combined_df), 
        len(combined_df[combined_df["Agreement"] == "False Positive"])/len(combined_df), 
        len(combined_df[combined_df["Agreement"] == "False Negative"])/len(combined_df)
        ]

        plt.rcParams['font.size'] = 8.0
        fig2, ax = plt.subplots(figsize=(6,6))
        _,_,autotexts=ax.pie(agreement_ratio, 
        labels = [
            "Hatespeech and Insult", 
            "Neither Hatespeech nor Insult", 
            "Not Insult but Hatespeech", 
            "Insult but not Hatespeech"
            ],
            colors=['#ad9176', '#795548', '#00695c','#b71c1c'],
            explode=(0.05,0.05,0.05,0.05), 
            autopct='%1.1f%%'
            )

        for autotext in autotexts:
            autotext.set_color('white')
        # fig2.suptitle("Trumps twitter insults, labeled by the New York Times and our Hatespeech Model:")
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)

    st.write("---------------")

    st.markdown("## 2. Testing our Hatespeech model on the Offensive tweets dataset:")

    with st.expander("Click here to see the results:"):

        def load_file(file):
            with open(file, mode='r') as f:
                data = f.readlines()
                data = [i.strip("\n") for i in data]
            return data

        train_text = load_file("./streamlit/data/offensive/train_text-2.txt")
        train_labels = load_file("./streamlit/data/offensive/train_labels-2.txt")
        train_labels = [int(i) for i in train_labels]

        OffenseDF = pd.DataFrame()
        OffenseDF["Offense_Labels"] = train_labels
        OffenseDF["Tweets"] = train_text

        offense_predict = open_jar("./data/pickle/classification_pickles/OffensePrediction.pkl")

        OffenseDF["HS_Label"] = label_predictions(offense_predict)

        data = [len(OffenseDF[OffenseDF["Offense_Labels"] == 0])/len(OffenseDF),len(OffenseDF[OffenseDF["Offense_Labels"] == 1])/len(OffenseDF)]
        data2 = [len(OffenseDF[OffenseDF["HS_Label"] == 0])/len(OffenseDF), len(OffenseDF[OffenseDF["HS_Label"] == 1])/len(OffenseDF)]
        labels = ['Offensive', 'Not Offensive']
        labels2 = ['Hatespeech', "Not Hatespeech"]

        fig, ax = plt.subplots(1,2, figsize=(16,7))
        _,_,autotexts0=ax[0].pie(data, labels=labels, colors=['#00695c','#b71c1c'],explode=(0, 0.1), autopct='%1.1f%%', shadow=True)
        _,_,autotexts1=ax[1].pie(data2, labels=labels2, autopct='%1.1f%%',colors=['#00695c','#b71c1c'],explode=(0, 0.1), shadow=True)
        for autotext in autotexts0:
            autotext.set_color('white')
        for autotext in autotexts1:
            autotext.set_color('white')
        plt.tight_layout()
        fig.suptitle("Insult label ratio to hatespeech label ratio comparison:")
        st.pyplot(fig)

        OffenseDF = OffenseDF.reset_index(drop=True)

        def agreement_func():
            results = []
            for i in range(0, 11916):
                if OffenseDF["Offense_Labels"][i] == 1 and  OffenseDF["HS_Label"][i] == 1:
                    results.append("Agree Offense")
                elif OffenseDF["Offense_Labels"][i] == 0 and  OffenseDF["HS_Label"][i] == 0:
                    results.append("Agree Not Offense")
                elif OffenseDF["Offense_Labels"][i] == 1 and OffenseDF["HS_Label"][i] == 0:
                    results.append("False Negative")
                elif OffenseDF["Offense_Labels"][i] == 0 and OffenseDF["HS_Label"][i] == 1:
                    results.append("False Positive")
            return results

        OffenseDF["Agreement"] = agreement_func()
        
        agreement_ratio = [
        len(OffenseDF[OffenseDF["Agreement"] == "Agree Offense"])/len(OffenseDF),
        len(OffenseDF[OffenseDF["Agreement"] == "Agree Not Offense"])/len(OffenseDF), 
        len(OffenseDF[OffenseDF["Agreement"] == "False Positive"])/len(OffenseDF), 
        len(OffenseDF[OffenseDF["Agreement"] == "False Negative"])/len(OffenseDF)
        ]
        fig, ax = plt.subplots(figsize=(16,7))
        _,_,autotexts=ax.pie(agreement_ratio, labels = ["Offensive and Hatespeech", "Neither Offensive nor Hatespeech", "Offensive but not Hatespeech", "Not Offensive but Hatespeech"],colors=['#795548','#ad9176','#00695c','#b71c1c'],explode=(0.05,0.05,0.05,0.05), autopct='%1.1f%%')
        for autotext in autotexts:
            autotext.set_color('white')
        fig.suptitle("Offensive labeling to Hatespeech classification agreement ratio:")
        plt.tight_layout()

        st.pyplot()

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

    st.sidebar.title("Checkout the following:")
    st.sidebar.markdown("""
    - _Start Page_
        - Introduction to the project
    - _Preprocessing_
        - The datasets we used
        - Our tokenizers
        - A comparison of these on the hatespeech dataset
    - _Data Characterisation_
        - Corpora Statistics
        - Most frequent tokens
    - _Manual Annotation_
        - Group Manual Annotation results
        - Survey Manual Annotation results
        - Comparisons of these
    - _Automatic Prediction_
        - Comparisons between different machine learning models used
    - _Data Augmentation_
        - Interactive classification
        - **Labeling Trump's twitter insults**
        - Hatespeech and Offensive tweet classification comparison
    """)
    st.sidebar.write("-----------------")

    
    if mode_two == sidebar_options[0]:
        start_page()

    elif mode_two == sidebar_options[1]:
        preprocessing()

    elif mode_two == sidebar_options[2]:
        data_char()

    elif mode_two == sidebar_options[3]:
        man_anot()

    elif mode_two == sidebar_options[4]:
        auto_predic()

    elif mode_two == sidebar_options[5]:
        data_aug()



if __name__ == "__main__":
    main()