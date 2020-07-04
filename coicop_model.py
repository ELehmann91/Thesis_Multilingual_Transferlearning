from tqdm import tqdm
import re 
import pickle
import pandas as pd
import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import random
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler

# load model
lstm_model = tf.keras.models.load_model('de_fr_mod_cc5.h5')

with open('coicop_5_4.txt') as json_file:#
    coicop_5_4 = json.load(json_file)

with open('coicop_5_3.txt') as json_file:#
    coicop_5_3 = json.load(json_file)

class predictor:
    '''
    Takes the text and makes beautiful predictions for coicop categories
    '''
    def __init__(self,df,name_col,cat_col,url_col,label_col,lang,embedding_dim=300,label_dict3=coicop_5_3,label_dict4=coicop_5_4
                 ,model=lstm_model,batchsize=100):
        self.embedding_dim = embedding_dim
        self.df = df
        self.label_col = label_col
        if url_col is not None:
            self.df['url_text'] = self.df[url_col].fillna('unknown').apply(lambda x: (self.parse_url(x)))

        if cat_col is not None and url_col is not None:
            self.df['text'] = self.df[name_col].fillna('unknown') + ' <sep> ' + self.df[cat_col].fillna('unknown') + ' <sep> ' + self.df['url_text']
            print('using name, category and words in url as input')
        elif cat_col is not None:
            self.df['text'] = self.df[name_col].fillna('unknown') + ' <sep> ' + self.df[cat_col].fillna('unknown') 
            print('using name and category as input')
        elif url_col is not None:
            self.df['text'] = self.df[name_col].fillna('unknown') + ' <sep> ' + self.df['url_text'] 
            print('using name and words in url as input')
        else:
            self.df['text'] = self.df[name_col].fillna('unknown')
            print('using only name as input')
        self.lang = lang
        if self.lang == 'fr':
            print('using french embeddings')
            self.emb = pickle.load(open('data/fr_muse_align.p', "rb" ) )
        if self.lang == 'de':
            print('using german embeddings')
            self.emb = pickle.load(open('data/de_slim_embed_ext.p', "rb" ) )  
        if self.lang == 'it':
            print('using italian embeddings')
            self.emb = pickle.load(open('data/it_slim_embed_ext.p', "rb" ) )  
        self.v = np.zeros(300)
        self.v[0]=1
        self.emb['<sep>'] = self.v
        self.model = model
        self.label_dict3 = label_dict3
        self.label_dict4 = label_dict4
        self.seq_len = int(np.quantile(df['text'].apply(lambda x: len(x.split())),.95))
        print('95% quantile no. of words per row is',self.seq_len,'(trained on 39)')
        self.labels5 = list(self.label_dict3.keys())[:74]
        self.labels5.append('9999_Non-Food')
        self.labels5.sort()
        self.batch=batchsize
        self.total = len(self.df)
        self.df['cc3_pred'] = None
        self.df['cc4_pred'] = None
        self.df['cc5_pred'] = None
        self.rep_dict = {'.':' ',
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

    def parse_url(self,url):
        url_list = str(url).split('/')[3:]
        url_str = ' '.join(w for w in url_list).lower()
        url_str = re.sub('[^a-zäöüàáâéèêßœ]+', ' ', url_str)
        url_str = ' '.join(w for w in url_str.split() if len(w)>2)
        return url_str
            
    def prepro(self,line):
        if isinstance(line,str):
            text_str = ' '.join(str(t) for t in line.split())
            text_str = text_str.lower()
            for a,b in self.rep_dict.items():
                text_str = text_str.replace(a,b)
            text_str = re.sub('[^a-zäöüàáâéèêßœ<>]+', ' ', text_str)
        else: 
            text_str = str(line)
            #print(line)
        return text_str

    def t2s(self,line):
        #tokens = []
        sen_embed = np.zeros((self.embedding_dim,self.seq_len))
        line = self.prepro(line)
        words = line.split()
        for w in range(0,self.seq_len):
            try: 
                emb = self.emb[words[w]]
            except:
                emb = np.zeros(self.embedding_dim)
            #tokens.append(tok)
            sen_embed[:,w] = emb
        sen_embed = np.swapaxes(sen_embed,0,1)
        return sen_embed #np.array(tokens)

    def emb_to_pred(self,embeded,prd=False):
        y_pred5 = self.model.predict(embeded)
        y_pred5_arg = y_pred5.argmax(axis=1)
        y_pred5_max = y_pred5.max(axis=1)
        y_pr_lab5 = [self.labels5[y] for y in y_pred5_arg]
        y_pr_lab3 = [self.label_dict3[cc5] for cc5 in y_pr_lab5]
        y_pr_lab4 = [self.label_dict4[cc5] for cc5 in y_pr_lab5]
        if prd:
            return y_pr_lab3,y_pr_lab4,y_pr_lab5, y_pred5, y_pred5_max
        else:
            return y_pr_lab3,y_pr_lab4,y_pr_lab5

    def predict(self):
        resid = self.total % self.batch
        epochs = self.total // self.batch
        for i in tqdm(range(0,epochs)):
            fr_ = i*self.batch
            to_ = (i+1)*self.batch + (i+1==epochs) * resid
            text = self.df['text'][fr_:to_]
            #text_pre = [str(self.prepro(t)) for t in text]
            #print(text_pre)
            text_emb = np.array([self.t2s(t) for t in text])
            text_prd = self.emb_to_pred(text_emb)
            self.df['cc3_pred'][fr_:to_] = text_prd[0]
            self.df['cc4_pred'][fr_:to_] = text_prd[1]
            self.df['cc5_pred'][fr_:to_] = text_prd[2]
        print('predictions ready')    
                                                    
    def test_performance(self):
        df_acc = self.df[self.df[self.label_col].isna()==False]
        acc = round(accuracy_score(df_acc['cc5_pred'],df_acc[self.label_col]),4) *100
        print('number of observation (labeled / all):',len(df_acc),'/',len(self.df),'consistency ',acc,'%')

    def confusion_matrix(self):
        df_acc = self.df[self.df[self.label_col].isna()==False]
        print(classification_report(df_acc['cc5_pred'], df_acc[self.label_col]))
        
                
    def predict_proba(self):
        prediction = []
        max_score = []
        resid = self.total % self.batch
        epochs = self.total // self.batch
        for i in tqdm(range(0,epochs)):
            fr_ = i*self.batch
            to_ = (i+1)*self.batch + (i+1==epochs) * resid
            text = self.df['text'][fr_:to_]
            #text_pre = [str(self.prepro(t)) for t in text]
            #print(text_pre)
            text_emb = np.array([self.t2s(t) for t in text])
            text_prd = self.emb_to_pred(text_emb,True)
            self.df['cc3_pred'][fr_:to_] = text_prd[0]
            self.df['cc4_pred'][fr_:to_] = text_prd[1]
            self.df['cc5_pred'][fr_:to_] = text_prd[2]
            #probabilities
            #y_pred5 = self.model.predict(text_emb)
            prediction.extend(text_prd[3])
            max_score.extend(text_prd[4])
        df_probs = pd.DataFrame(prediction,columns=self.labels5,index=self.df.index) 
        df_probs['max_score'] = max_score
        
        for col in df_probs.columns:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
                
        df_comb = self.df.join(df_probs)
        return df_comb
    
    def single_pred(self,input_t):
        if type(input_t) == 'str':
            emb = np.array(self.t2s(input_t))
            return self.model.predict(np.expand_dims(emb,axis=0))
        else:
            try:
                emb = np.array([self.t2s(t) for t in input_t])
                return new_model.predict(emb)
            except:
                print('ErROr eRRoR')
        
    def get_df(self):
        return self.df

    def get_predict_function(self):
        def predict_func(texts):
            text_emb = np.array([self.t2s(t) for t in texts])
            y_pred5 = self.model.predict(text_emb)
            return y_pred5[:,:]
        return predict_func
    
    def tell_me_why(self,text=None,categ=None):
        if text is None:
            if categ is not None:
                if len(self.df[self.df['cc5_pred']==categ]) == 0:
                    print('no prediction for',categ)
                    
                n = random.randint(0, len(self.df[self.df['cc5_pred']==categ]))
                text = self.df['text'][self.df['cc5_pred']==categ].iloc[n]
                print('prediction',self.df['cc5_pred'][self.df['cc5_pred']==categ].iloc[n])
                if self.label_col is not None and self.label_col in self.df.columns:
                    print('label',self.df[self.label_col][self.df['cc5_pred']==categ].iloc[n])
            else:
                n = random.randint(0, len(self.df))
                text = self.df['text'].iloc[n]
                print('prediction',self.df['cc5_pred'].iloc[n])
                if self.label_col is not None and self.label_col in self.df.columns:
                    print('label',self.df[self.label_col].iloc[n])
        predict_func = self.get_predict_function()
        sampler = MaskingTextSampler(replacement="UNK", max_replace=0.7, token_pattern=None, bow=False)
        te = TextExplainer(sampler=sampler, position_dependent=True, random_state=42)
        te.fit(text, predict_func)
        return  te.explain_prediction(target_names=self.labels5, top_targets=3) 
    
