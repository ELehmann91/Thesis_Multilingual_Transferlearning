
import ipywidgets as widgets
from IPython.display import display
import random 
from  ipywidgets import Layout, HBox, VBox, Box
import pandas as pd
import json
import sys 
pd.set_option('mode.chained_assignment', None)

# get dict
path ='/content/gdrive/My Drive/Thesis_ecb_ecoicop'

with open(path+'/data/coicop_5_4.txt') as json_file:#
    coicop_5_4 = json.load(json_file)
with open(path+'/data/coicop_5_3.txt') as json_file:#
    coicop_5_3 = json.load(json_file)
    
class labeler:
    '''
    This object takes a dataframe of web scraped product data as input and displays the prediction of unlabeld data in a dropdown. 
    The user can accept or edit the prediction and save the label.
    '''
    def __init__(self,labeled_by, df,text1,text2,url_str,coicop_5_4=coicop_5_4,coicop_5_3=coicop_5_3):
        self.counter=0
        self.labeled_by = labeled_by
        self.text1 = text1
        self.text2 = text2
        self.url_str = url_str
        self.df = df.reset_index(drop=True) 
        self.order()
        if 'labeled_by' not in df.columns:
            self.df['labeled_by'] = None
        self.labels5 = list(coicop_5_4.keys())
        self.labels5.sort()
        if 'cc5' in df.columns:
            self.df_idx =  list(self.df.index[self.df['cc5'].isna()])
        else:
            self.df['cc3'] = None
            self.df['cc4'] = None
            self.df['cc5'] = None
            self.df_idx =  list(range(0,len(self.df)))

        self.idx =  self.df_idx[0]
        try:
            self.text_orig = self.df[self.text1].loc[self.idx]
            self.text_trans = self.df[self.text2].loc[self.idx]
            self.cc5 = self.df['cc5_pred'].loc[self.idx]
            self.link = self.df[self.url_str].loc[self.idx]
        except:
            sys.exit('specify column names / dataframe needs columns: "text_orig","text_trans","cc3_pred","cc4_pred","cc5_pred","url"')
            

    def order(self):
        toless = list(self.df['cc5'].value_counts()[-10:].index)
        not_labeled = [lab for lab in self.df['cc5_pred'].value_counts().index if lab not in self.df['cc5'].value_counts().index]
        toless.extend(not_labeled)
        self.df['sort_columns'] = 0

        toless = self.df['cc5'].value_counts()[-10:].index
        for i,cc in enumerate(toless):
            self.df['sort_columns'][self.df['cc5_pred']==cc] = i   

        self.df = self.df.sort_values('sort_columns',ascending=False) 

    def get_stats(self):
        len_df = len(self.df)
        new_lab =  (len_df -len(self.df_idx) - len(self.df[self.df['cc5'].isna()==False])) *-1
        labeled = len(self.df[self.df['cc5'].isna()==False])
        print('new labels:',new_lab)
        print('in total',labeled,'of',len_df,'labeled (',round(labeled/len_df,2)*100,'%)')

    def init_widget(self):
        self.text_widget1 = widgets.Text(value=self.text_orig, description='Text 1:',layout=Layout(width='80%', height='40px'), disabled=False)
        self.text_widget2 = widgets.Text(value=self.text_trans, description='Text 2:',layout=Layout(width='80%', height='40px'), disabled=False)
        self.dd_cc5 = widgets.Dropdown(options=self.labels5, value=self.cc5, description='COICOP 5:',layout=Layout(width='80%', height='40px'),  disabled=False)
        self.dd_cc5_out = widgets.Output()
        self.button_save = widgets.Button(description="Save")
        self.output_save = widgets.Output()
        self.button_next = widgets.Button(description="Next")
        self.output_next = widgets.Output()
        self.button_link = widgets.Button(description="GetLink")
        self.output_link = widgets.Output()

    def pick_obs(self):
        self.counter += 1
        self.idx =  self.df_idx[self.counter]
        self.text_orig = self.df[self.text1].loc[self.idx]
        self.text_trans = self.df[self.text2].loc[self.idx]
        self.cc5 = self.df['cc5_pred'].loc[self.idx]
        self.link = self.df[self.url_str].loc[self.idx]
        self.text_widget1.value = self.text_orig
        self.text_widget2.value = self.text_trans
        self.dd_cc5.value = self.cc5


    def display_all(self):

        items_auto = [self.dd_cc5,self.dd_cc5_out]
        items_0 = [self.button_next, self.output_next,self.button_save, self.output_save,self.button_link, self.output_link]
        box_auto = Box(children=items_auto, layout=Layout(display='flex',flex_flow='column', align_items='stretch', align_content='center', width='80%'))
        box_0 = Box(children=items_0, layout=Layout(display='flex',flex_flow='row', align_items='stretch', align_content='center', width='80%'))
        display(VBox([self.text_widget1,self.text_widget2,box_auto, box_0]))
      
    def output_labels(self):

        return self.df

    def start_to_label(self):
        self.init_widget()
        self.display_all()

        def on_value5_change(change):
            with self.dd_cc5_out:
                self.cc5 = change['new']
            self.dd_cc5_out.clear_output()
              
        def on_button_save_clicked(b):
            with self.output_save:
                self.df['cc3'].loc[self.idx] = coicop_5_3[self.cc5]
                self.df['cc4'].loc[self.idx] = coicop_5_4[self.cc5]
                self.df['cc5'].loc[self.idx] = self.cc5
                self.df['labeled_by'].loc[self.idx] = self.labeled_by
            self.pick_obs()

        def on_button_next_clicked(b):
            self.pick_obs()

        def on_button_link_clicked(b):
            print(self.link)

        self.dd_cc5.observe(on_value5_change, names='value')
        self.button_save.on_click(on_button_save_clicked)
        self.button_next.on_click(on_button_next_clicked)
        self.button_link.on_click(on_button_link_clicked)
