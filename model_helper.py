from tqdm import tqdm 
import numpy as np
import re
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def balanced_train_test_split(X,y,by):
    X_train, X_val_test, y_train, y_val_test = train_test_split(X
                                                              , y
                                                              , test_size=0.3
                                                              , random_state=99
                                                              , shuffle=True
                                                              , stratify=y[by])
    X_val, X_test, y_val, y_test = train_test_split(X_val_test
                                                  , y_val_test
                                                  , test_size=0.50
                                                  , random_state=99
                                                  , stratify=y_val_test[by])
    
    df_train = pd.concat([X_train,y_train], axis=1)
    df_val = pd.concat([X_val,y_val], axis=1)
    df_test = pd.concat([X_test,y_test], axis=1)

    # upsample
    max_cat_cnt = df_train[by].value_counts()[0]
    for categ in df_train[by].unique():
        df_sample = df_train[df_train[by]==categ]
        df_train = df_train[df_train[by]!=categ]
        no_ = len(df_sample)
        df_minority_upsampled = resample(df_sample, 
                                      replace=True,     # sample with replacement
                                      n_samples=max_cat_cnt,    # to match majority class
                                      random_state=123) # reproducible results
        df_train = pd.concat([df_train, df_minority_upsampled])
    print(df_train.shape,df_val.shape,df_test.shape)
    df_train.reset_index(drop=True)
    return df_train, df_val, df_test


rep_dict = {'.':' ',
                ',': ' ',
                '&': ' ',
                '-': ' ',
                '/': ' ',
                '%': ' percent ',
                '1': ' one ',
                '2': ' two ',
                '3': ' three ',
                '4': ' four ',
                '5': ' five ',
                '6': ' six ',
                '7': ' seven ',
                '8': ' eigth ',
                '9': ' nine ',
                '0': ' zero ',
                ' l ':' liter ',
                ' ml ':' liter '
                }
                
class text_to_embed(object):
    '''
    takes text and embeddingmodel as input and outputs sequence of embeddings
    '''
    def __init__(self
                 , text = None
                 , lang = None
                 , embed_de = None
                 , embed_fr = None
                 , seq_len = None
                 , rep_dict = rep_dict
                 , embedding_dim=300):
                 
        self.text = text
        self.lang = lang
        self.embed_de = embed_de
        self.embed_fr = embed_fr
        self.seq_len = seq_len
        self.rep_dict = rep_dict
        self.embedding_dim = embedding_dim

    def prepro(self,line):
        text_str = ' '.join(t for t in line.split())
        text_str = text_str.lower()
        for a,b in self.rep_dict.items():
            text_str = text_str.replace(a,b)
        text_str = re.sub('[^a-zäöüàáéèêß]+', ' ', text_str)
        return text_str

    def t2s(self,line,la):
        #tokens = []
        sen_embed = np.zeros((self.embedding_dim,self.seq_len))
        words = line.split()
        for w in range(0,self.seq_len):
            try: 
              if la == 'de':
                  emb = self.embed_de[words[w]]
              elif la == 'fr':
                  emb = self.embed_fr[words[w]]
            except:
              emb = np.zeros(self.embedding_dim)
            #tokens.append(tok)
            sen_embed[:,w] = emb

        sen_embed = np.swapaxes(sen_embed,0,1)
        return sen_embed #np.array(tokens)

    def __iter__(self):
        for line,la in tqdm(zip(self.text,self.lang)):
            line = self.prepro(line)
            line = self.t2s(line,la)
            yield line
            

path ='/content/gdrive/My Drive/Thesis_ecb_ecoicop'
with open(path+'/data/label_cc_dict.json') as json_file:#
    label_cc_dict = json.load(json_file)

# Recreate the exact same model purely from the file
new_model = tf.keras.models.load_model(path+'/model/ger_pool_wo_emb.h5')

def text_to_model_input(embeded,model=new_model,label_dict=label_cc_dict):
    labels3 = label_dict['cc3']
    labels4 = label_dict['cc4']
    labels5 = label_dict['cc5']
    labels3.sort()
    labels4.sort()
    labels5.sort()

    y_pred3,y_pred4,y_pred5 = model.predict(embeded)
    y_pred3_arg = y_pred3.argmax(axis=1)
    y_pr_lab3 = [labels3[y] for y in y_pred3_arg]
    y_pred4_arg = y_pred4.argmax(axis=1)
    y_pr_lab4 = [labels4[y] for y in y_pred4_arg]
    y_pred5_arg = y_pred5.argmax(axis=1)
    y_pr_lab5 = [labels5[y] for y in y_pred5_arg]
    return y_pr_lab3,y_pr_lab4,y_pr_lab5
