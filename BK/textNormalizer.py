import contractions
from bs4 import BeautifulSoup
import unicodedata
import re
import nltk
default_st = nltk.sent_tokenize

class preProcessing:
    '''The following functions help us build our text-wrangling system.'''
    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub('r[\r|\n|\r\n]+', '\n', stripped_text)
        return stripped_text

    def customized_pre_process(self, text):
        tmp_x = text
        pattern = "\r\n[ ]*"
        tmp_x = re.subn(pattern, "", tmp_x)
        tmp_x = tmp_x[0]
        pattern = u"\xa0"
        tmp_x = re.subn(pattern, "", tmp_x)
        tmp_x = tmp_x[0]
        text_sentences = default_st(text=tmp_x)
        return text_sentences

    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def expand_contractions(self, text):
        return contractions.fix(text)

    def remove_special_characters(self, text, remove_digits=False):
        pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
        text = re.sub(pattern, '', text)
        return text

    def pre_process_document(self, document):
        # strip HTML
        document = self.strip_html_tags(document)

        # customized pre processing
        document = self.customized_pre_process(document)

        # process every sentence
        for i in range(len(document)):
            sentence = document[i]
            # lower case
            sentence = sentence.lower()
            # remove accented characters
            sentence = self.remove_accented_chars(sentence)
            #remove special characters and\or digits
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            sentence = special_char_pattern.sub(" \\1", sentence)
            sentence = self.remove_special_characters(sentence, remove_digits=True)
            # remove extra whitespace
            sentence = re.sub(' +', ' ', sentence)
            document[i] = sentence

        return document

tn_preProcessing = preProcessing()