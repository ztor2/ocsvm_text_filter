import json
import re
import pickle
import joblib
import pathlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import ckonlpy
from ckonlpy.tag import Twitter
from ckonlpy.tag import Postprocessor
import warnings; warnings.filterwarnings('ignore')

class ocsvm_text_filter:
    def __init__(self, text_data_name, model_name):
        self.text_data_name = text_data_name
        self.model_name = model_name
        
    def preprocessing(self):
        with open(self.text_data_name, 'r', encoding='utf-8') as f:
            text_json = json.load(f, strict=False)
        if 'review_text' in text_json[0].keys():
            text = [self.del_nonkr(text_json[i]['review_text']) for i in range(len(text_json))]
        else:
            text = [self.del_nonkr(text_json[i]['cmt_body']) for i in range(len(text_json))]
        corp = self.tokenize_text(text)
        return text_json, corp
    
    def get_noise_idx(self, corp):
        emb_matrix = self.get_emb_with_vocabfile(corp)
        ocsvm_classifier = joblib.load(self.model_name)
        print('Model: {}'.format(ocsvm_classifier))
        result = ocsvm_classifier.predict(emb_matrix)
        noise_idx = np.where(result == -1)[0]
        print('{} noise texts are detected.'.format(len(noise_idx)))
        print('{} texts will be saved.'.format(len(corp)-len(noise_idx)))
        return noise_idx
    
    def save_filtered_text(self, text_json, noise_idx, new_data_name):
        self.del_elements(text_json, noise_idx)
        with open(new_data_name, 'w') as f:
            json.dump(text_json, f)
        return print('Filtered text data saved as {}'.format(str(pathlib.Path.cwd()) + '\\' + new_data_name))

    def del_nonkr(self, i):
        i = re.compile('[^\d \u3131-\u3163\uac00-\ud7a3]+').sub('', i)
        i = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+').sub('', i)
        i = re.sub(r'([ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥]+)', ' ', i)
        i = i.replace('_', ' ')
        i = i.strip()
        return i
    
    def tokenize_text(self, text):
        twitter = ckonlpy.tag.Twitter()
        tokenizer = Postprocessor(base_tagger=twitter)
        with open('dictionary.pkl', 'rb') as f:
            dict_toadd = pickle.load(f)
        for i in dict_toadd:
            twitter.add_dictionary(i,'Noun')
        with open('stopwords.pkl', 'rb') as f:
            stopwords = pickle.load(f)
        
        tokenized_corpus = []
        for sentence in text:
            tmp = [i[0] for i in  tokenizer.pos(sentence)  if not i[0] in stopwords] 
            joined = ' '.join(tmp)
            if len(joined) > 1:
                tokenized_corpus.append(joined)
        return tokenized_corpus
    
    def get_emb_with_vocabfile(self, corp):
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        emb = vectorizer.fit_transform(corp)
        return emb
    
    def del_elements(self, list_obj, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_obj):
                list_obj.pop(idx)