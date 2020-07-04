from tqdm import tqdm 
import numpy as np
import re
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

path ='/content/gdrive/My Drive/Thesis_ecb_ecoicop'
with open(path+'/data/ecoicop_json.txt') as json_file:#
    coicop_dic = json.load(json_file)
    
class prepare_df(object):
    '''
    takes dataframe and columns names and outputs  standardized dataframe
    '''
    def __init__(self, df_in = None, lang = None, name = None, categ = None, prod_desc = None, text_other = None
                 , url = None, unit = None, cc3 = None, cc4 = None, cc5 = None, cc3_pred = None, cc4_pred = None, cc5_pred = None
                 , shop = None, brand=None, price=None, id = None, labeld_by = None, coicop_dic = {}):
        self.df_in = df_in
        self.df_in['lang'] = lang
        self.lang = 'lang'
        self.name = name
        self.categ = categ
        self.prod_desc = prod_desc
        self.text_other = text_other
        self.url = url
        self.unit = unit
        self.cc3 = cc3
        self.cc4 = cc4
        self.cc5 = cc5
        self.cc3_pred = cc3_pred 
        self.cc4_pred = cc4_pred 
        self.cc5_pred = cc5_pred 
        self.shop = shop
        self.brand = brand
        self.price = price
        self.id = id
        self.labeld_by = labeld_by
        self.coicop_dic = coicop_dic
        self.rep_dict = {'.':' ', ',': ' ', '&': ' ', '-': ' ', '/': ' ','|':' '  }

    def parse_url(self,url):
        url_list = str(url).split('/')[3:]
        url_str = ' '.join(w for w in url_list).lower()
        for a,b in self.rep_dict.items():
            url_str = url_str.replace(a,b)
        url_str = re.sub('[^a-zäöüàáâéèêßœ]+', ' ', url_str)
        url_str = ' '.join(w for w in url_str.split() if len(w)>2)
        return url_str
    
    def fill_frame(self):
        df_out = pd.DataFrame()
        for attr, value in self.__dict__.items():
            if value is None:
                df_out[attr] =                  None
            elif isinstance(value, str):
                if attr in ['cc3','cc4','cc5']:
                    if len(self.coicop_dic) == 0:
                        df_out[attr] =              self.df_in[value]
                    else:
                        df_out[attr] =              self.df_in[value].apply(lambda x: '999' if np.isnan(x) else str(int(x))).map(self.coicop_dic)
                elif attr == 'url':
                    df_out[attr] =              self.df_in[value].fillna('unknown')
                    df_out['words_from_url'] =  self.df_in[value].apply(lambda x: self.parse_url(x)).fillna('unknown')
                elif attr == 'categ':
                    df_out[attr] =              self.df_in[value].apply(lambda x: str(x).replace('|',' ').replace('/',' ')).fillna('unknown')
                else: 
                    df_out[attr] =              self.df_in[value].fillna('unknown')

        return df_out
        
def prepro(line,rep_dict):
    if isinstance(line,str):
        text_str = ' '.join(str(t) for t in line.split())
        text_str = text_str.lower()
        for a,b in rep_dict.items():
            text_str = text_str.replace(a,b)
        text_str = re.sub('[^a-zäöüàáâéèêßœ]+', ' ', text_str)
    else: 
        text_str = str(line)
        print(line)
    return text_str

class vocab(object):
    '''
    takes the standardized dataframe and gives out vocab, with index, with word counts
    the text in rows as list of string or list of token and
    builds a subset of embedding including out ov vocabulary items
    '''
    def __init__(self, df_in = None):
        self.df_in = df_in
        self.lang = str(self.df_in['lang'].iloc[1])
        self.df_in['text'] =  self.df_in['name'].fillna('unknown') + ' ' + \
                              self.df_in['categ'].fillna('unknown') + ' ' + \
                              self.df_in['prod_desc'].fillna('unknown') + ' ' + \
                              self.df_in['words_from_url'].fillna('unknown') 
    
    def get_list(self,preprocess=True,token=False,one_obj =False):
        if preprocess:
            list_str = [prepro(line,rep_dict) for line in self.df_in['text']]
        else:
            list_str = [line for line in self.df_in['text']]
        if token:
            list_str = [[word for word in line.split()] for line in list_str]
        if one_obj:
            temp_list = []
            for line in list_str:
                tokens = [word for word in line.split()]
                temp_list.extend(tokens)
            list_str = temp_list
        return list_str

    def get_vocab(self,index=False,count=False):
        vocab = {}
        i=1
        for tok in self.get_list(one_obj=True):
            if tok in vocab and count:
                vocab[tok] += 1
            else:
                if count:
                    vocab[tok] = 1
                else:
                    vocab[tok] = i
                    i += 1
        if index or count:
            return {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1],reverse=True)}
        else: 
            return list(vocab.keys())

    def slim_embed(self,ooV=True):
        print('lean back, this takes a while')
        if self.lang == 'fr':
            embed = KeyedVectors.load_word2vec_format('/content/gdrive/My Drive/Thesis_ecb_ecoicop/embeddings/wiki.fr.vec')
        if self.lang == 'de':
            embed = KeyedVectors.load_word2vec_format('/content/gdrive/My Drive/Thesis_ecb_ecoicop/embeddings/wiki.de.vec')
        print('embedding loaded, length: ',len(embed.vocab))

        slim_embed = {}
        ooV = []
        for tok in self.get_vocab():
            if tok in embed:
                slim_embed[tok] = embed[tok]
            else:
                ooV.append(tok)
        print('embed slim',len(slim_embed),'out of vocav',len(ooV))
        if ooV:
            frq = self.get_vocab(count=True)
            oov_dict={}
            for tok in oov:
                oov_dict[tok] = frq[tok]
                oov_dict = {k: v for k, v in sorted(oov_dict.items(), key=lambda item: item[1], reverse=True)}
            return slim_embed, oov_dict
        else:
            return slim_embed

        
def balanced_train_test_split(X,y,by,n=.8):
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
                                      n_samples=int(max_cat_cnt*n),    # to match majority class
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
                'ü': 'ue',
                'ä': 'ae',
                'ö': 'oe',
                'ß': 'ss',
                'ê': 'e',
                'é': 'e',
                'è': 'e',
                'â': 'a',
                'á': 'a',
                'à': 'a',
                'œ': 'ae',
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
                 , embed = None
                 , seq_len = None
                 , rep_dict = rep_dict
                 , embedding_dim=300):
                 
        self.text = text
        self.lang = lang
        self.embed = embed
        self.v = np.zeros(300)
        self.v[0]=1
        self.embed_de['<sep>'] = self.v
        self.embed_fr['<sep>'] = self.v
        self.seq_len = seq_len
        self.rep_dict = rep_dict
        self.embedding_dim = embedding_dim

    def prepro(self,line):
        text_str = ' '.join(t for t in line.split())
        text_str = text_str.lower()
        for a,b in self.rep_dict.items():
            text_str = text_str.replace(a,b)
        text_str = re.sub('[^a-zäöüàáâéèêßœ<>]+', ' ', text_str)
        return text_str

    def t2s(self,line,la):
        #tokens = []
        sen_embed = np.zeros((self.embedding_dim,self.seq_len))
        words = line.split()
        for w in range(0,self.seq_len):
            try: 
                emb = self.embed[words[w]]
            except:
                emb = np.random.normal(self.embedding_dim)
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
new_model = tf.keras.models.load_model(path+'/model/de_fr_mod_cc5.h5')

def text_to_model_input(embeded,model=new_model,label_dict=label_cc_dict):

    labels3 = label_dict['cc3']
    labels4 = label_dict['cc4']
    labels5 = label_dict['cc5']
    labels3.append('99_Non-Food')       
    labels4.append('999_Non-Food')
    labels5.append('9999_Non-Food')
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
