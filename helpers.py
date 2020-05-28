#import all needed libraries 

import pandas as pd
import numpy as np
import keras 
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re 
from keras import backend as K



def data_preprocessing(text):
    """
    Data preprocessing: cleaning, removing emojis, numbers etc.
    
    @text:str, text of one tweet

    @return: str, the preprocessed text
    """
    def clean_text(text):
        """
        Cleaning the text
        @text:str, text of one tweet
        @return: str, the cleaned text
        """
        text = text.lower()
        cleaned_text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
        
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        cleaned_text = emoji_pattern.sub(r'', cleaned_text)
        return cleaned_text

    def remove_numbers(text):
        """
        Removing numbers 
        @text:str, text of one tweet
        @return: str, text without numbers
        """
        text_no_num = re.sub(r'\d+', '', text)
        return text_no_num
    
    def remove_whitespaces(text):
        """
        Removing whitespaces
        @text:str, text of one tweet
        @return: str, text without whitespaces
        """
        text_no_spaces = " ".join(text.split())
        return text_no_spaces
    
    def remove_stopwords(text):
        """
        Removing stopwords
        @text:str, text of one tweet
        @return: str, text without stopwords
        """
        stop_words = set(stopwords.words("english")) 
        tokens = word_tokenize(text)
        tokens_no_stop = [word for word in tokens if word not in stop_words]
        return tokens_no_stop
    
    def combine_text(text):
        """
        Removing stopwords
        @text:str, text of one tweet ["word1", "word2", "word3"]
        @return: str, combined text
        """
        combined_text = ' '.join(text)
        
        return combined_text

    text = clean_text(text)
    text = remove_numbers(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = combine_text(text)
    
    return text

def fix_labels(data):
    """
    Fix labels. Some of the labels are incorrect 
    (https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert#5.-Mislabeled-Samples)
    
    @text:DataFrame, dataset that contains all tweets 
    @return: DataFrame, dataframe with new column that contains fixed labels
    """
    
    data['target_fixed'] = data['target'].copy() 

    data.loc[data['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_fixed'] = 0
    data.loc[data['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_fixed'] = 0
    data.loc[data['text'] == 'To fight bioterrorism sir.', 'target_fixed'] = 0
    data.loc[data['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_fixed'] = 1
    data.loc[data['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_fixed'] = 1
    data.loc[data['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_fixed'] = 0
    data.loc[data['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_fixed'] = 0
    data.loc[data['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_fixed'] = 1
    data.loc[data['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_fixed'] = 1
    data.loc[data['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_fixed'] = 0
    data.loc[data['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_fixed'] = 0
    data.loc[data['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_fixed'] = 0
    data.loc[data['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_fixed'] = 0
    data.loc[data['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_fixed'] = 0
    data.loc[data['text'] == "Caution: breathing may be hazardous to your health.", 'target_fixed'] = 1
    data.loc[data['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_fixed'] = 0
    data.loc[data['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_fixed'] = 0
    data.loc[data['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_fixed'] = 0
    
    return data

def recall_m(y_true, y_pred):
    """
    Function to calculate recall and use it as one of the metrics in model.compile(...). 
    
    @y_true: float, ground truth label of the tweet
    @y_pred: float, predicted label of the tweet
    @return: float, recall
    """   
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    """
    Function to calculate precision and use it as one of the metrics in model.compile(...). 
    
    @y_true: float, ground truth label of the tweet
    @y_pred: float, predicted label of the tweet
    @return: float, precision
    """   
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    """
    Function to calculate F1 score and use it as one of the metrics in model.compile(...). 
    
    @y_true: float, ground truth label of the tweet
    @y_pred: float, predicted label of the tweet
    @return: float, F1 score
    """   
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
