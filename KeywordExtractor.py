import pandas as pd
import numpy as np
import yake
import json

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from keybert import KeyBERT


class KeywordExtractor:

    def __init__(self):
        """
           Class to extract and generate keywords from the text/transcription file.
        """
 
        self.stop_words = setup_stopwords()
        self.ModelDict = {'KeyBERT': KeyBERT(model='all-mpnet-base-v2')}
       
        self.KeyBertModel = KeyBERT(model='all-mpnet-base-v2')
        self.top_num_words = 55
     
        self.complete_string = ""

    def extract_keywords(self, transcription_file, top_num_words):
        """
        Function to extract the keywords from the transcription file. Will extract upto the provided
        top_num_words number.
        :param transcription_file: JSON file of recorded transcriptions
        :param top_num_words: Top 'n' words to be extracted.
        :return: List of top n keywords for a document with their respective distances to the input document.
        """
        self.complete_string = self.read_json(transcription_file)
    
        extracted_keywords = self.KeyBertModel.extract_keywords(self.complete_string,
                                                                keyphrase_ngram_range=(0, 1),
                                                                highlight=False,
                                                                top_n=int(top_num_words),
                                                                stop_words=list(self.stop_words))

        word_lst = list(dict(extracted_keywords).keys())
        return word_lst

    def read_json(self, transcription_file):
        """
        Read the JSON transcription file using pandas and return the combined string.
        :param transcription_file: JSON file of recorded transcriptions
        :return: complete_string: A complete string with all the conversation joined.
        """

        json_df = pd.read_json(json.dumps(transcription_file))
        complete_string = " ".join(json_df["ref"])
        return complete_string

    def update_model(self, model_name='KeyBERT'):
        """
        Update the model to be used.
        :param model_name: 'KeyBERT'
        :return: model architecture
        """
        return self.ModelDict[model_name]


def setup_stopwords():
    stop_words = set(ENGLISH_STOP_WORDS)
    return stop_words

