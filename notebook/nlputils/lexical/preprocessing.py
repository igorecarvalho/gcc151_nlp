import nltk
import unidecode
import string
import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

from nltk.corpus import stopwords
stop_words = stopwords.words('portuguese')

class Preprocessing:

    def __init__(self):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()

    def remove_accents(self, text):
        return unidecode.unidecode(text)

    def remove_punctuation1(self, text):
        return text.translate(str.maketrans('','',string.punctuation))
    
    def remove_punctuation(self, text):
        no_punc_text = []
        for sentence in text:
            sentence = REPLACE_NO_SPACE.sub("", sentence)
            sentence = REPLACE_WITH_SPACE.sub(" ", sentence)
            no_punc_text.append(sentence)
        return no_punc_text

    def tokenize_sentences(self, text):
        sentences = self.sent_tokenizer.tokenize(text)
        return sentences

    def tokenize_words(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        return tokens

    def remove_stopwords(self, sentence):
        for word in sentence:
            #remocao de stopwords
            if word in stop_words:
                sentence.remove(word)
        return sentence

    def stemmize(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def lowercase(self, text):
        return text.lower()
