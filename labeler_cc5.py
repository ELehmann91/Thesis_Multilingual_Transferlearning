#import model_helper
import ipywidgets as widgets
from IPython.display import display
import random 
from  ipywidgets import Layout, HBox, VBox, Box
import pandas as pd
import json
import sys 
pd.set_option('mode.chained_assignment', None)

# get dict

with open('coicop_5_4.txt') as json_file:#
    coicop_5_4 = json.load(json_file)

with open('coicop_5_3.txt') as json_file:#
    coicop_5_3 = json.load(json_file)
    
class labeler:
    '''
    This object takes a dataframe of web scraped product data as input and displays the prediction of unlabeld data in a dropdown. 
    The user can accept or edit the prediction and save the label.
    '''
    def __init__(self,labeled_by, df,text1,text2,url_str,CoiCop_5_pred_col,use_probabilities=False,coicop_5_4=coicop_5_4,coicop_5_3=coicop_5_3):
        self.counter=0
        self.df = df
        self.cc5_pred_col = CoiCop_5_pred_col
        self.labeled_by = labeled_by
        self.use_probabilities = use_probabilities
        self.text1 = text1
        self.toless = None
        if text2 in self.df.columns:
            self.text2 = text2
            self.df[text2] = self.df[text2].fillna('unknown')
        else:
            self.text2 = text1
            print(text2 ,'not found')
        if url_str in self.df.columns:
            self.url_str = url_str
            self.df[url_str] = self.df[url_str].fillna('unknown')
        else:
            self.url_str = text1
            print(url_str ,'not found')

        if 'labeled_by' not in df.columns:
            self.df['labeled_by'] = None
        
        self.labels5 = list(coicop_5_4.keys())[:74]
        self.labels5.append('9999_Non-Food')
        self.labels5.sort()
        
        if 'cc5' in df.columns:
            self.sort_cat()
            self.cc5_ord = self.toless[0]
            if self.use_probabilities:
                self.order()
            else:
                self.order2()
            #self.df_idx =  list(self.df.index[(self.df[self.cc5_pred_col]==self.cc5_ord) & (self.df['cc5'].isna())])

            
        else:
            self.df['cc3'] = None
            self.df['cc4'] = None
            self.df['cc5'] = None
            self.df_idx =  list(range(0,len(self.df)))
        self.idx =  self.df_idx[0]
        try:
            self.text_orig = self.df[self.text1].loc[self.idx]
            self.text_trans = self.df[self.text2].loc[self.idx]
            self.cc5 = self.df[self.cc5_pred_col].loc[self.idx]
            self.link = self.df[self.url_str].loc[self.idx]
        except:
            sys.exit('specify column names / dataframe needs columns: "text_orig","text_trans","cc3_pred","cc4_pred","cc5_pred","url"')
            
    def sort_cat(self):
        
        if self.use_probabilities:
            not_labeled = [lab for lab in self.labels5 if lab not in self.df['cc5'].value_counts().index]
        else:
            not_labeled = [lab for lab in self.df[self.cc5_pred_col].value_counts().index if lab not in self.df['cc5'].value_counts().index]
        
        toless = list(self.df['cc5'].value_counts().index)#.sort(reverse=True)
        not_labeled.extend(toless)
        self.toless = not_labeled


    def order(self):
        self.df = self.df.sort_values([self.cc5_ord],ascending=False)
        self.df_idx = self.df[df['cc5'].isna()].index
        print(len(self.df_idx))
        if len(self.df_idx) == 0:
                print('everything labeled')
                self.df_idx =  list(range(0,len(self.df)))

    def order2(self):
        self.df_idx  = list(self.df.index[(self.df[self.cc5_pred_col]==self.cc5_ord) & (self.df['cc5'].isna())]) 
        self.counter=0
        if len(self.df_idx) == 0:
                print('everything labeled')
                self.df_idx =  list(range(0,len(self.df)))


    def get_stats(self):
        len_df = len(self.df)
        new_lab = (len_df -len(self.df_idx) - len(self.df[self.df['cc5'].isna()==False]))*-1
        labeled = len(self.df[self.df['cc5'].isna()==False])
        print('new labels:',new_lab)
        print('in total',labeled,'of',len_df,'labeled (',round(labeled/len_df,2)*100,'%)')

    def init_widget(self):
        self.dd_cc5_ord = widgets.Dropdown(options=self.toless, value=self.cc5_ord, description='Select category to label:',layout=Layout(width='50%', height='60px'),  disabled=False)
        self.dd_cc5_ord_out = widgets.Output()
        self.text_widget1 = widgets.Text(value=self.text_orig, description='Text 1:',layout=Layout(width='80%', height='40px'), disabled=False)
        self.text_widget2 = widgets.Text(value=self.text_trans, description='Text 2:',layout=Layout(width='80%', height='40px'), disabled=False)
        self.text_widget3 = widgets.Text(value=self.link, description='Url:',layout=Layout(width='80%', height='40px'), disabled=False)
        self.dd_cc5 = widgets.Dropdown(options=self.labels5, value=self.cc5, description='COICOP 5:',layout=Layout(width='80%', height='40px'),  disabled=False)
        self.dd_cc5_out = widgets.Output()
        self.button_save = widgets.Button(description="Save")
        self.output_save = widgets.Output()
        self.button_next = widgets.Button(description="Next")
        self.output_next = widgets.Output()

    def pick_obs(self):
        if self.counter < len(self.df_idx):
            self.idx =  self.df_idx[self.counter]
            self.text_orig = self.df[self.text1].loc[self.idx]
            self.text_trans = self.df[self.text2].loc[self.idx]
            self.cc5 = self.df[self.cc5_pred_col].loc[self.idx]
            self.link = self.df[self.url_str].loc[self.idx]
            self.text_widget1.value = self.text_orig
            self.text_widget2.value = self.text_trans
            self.text_widget3.value = self.link
            self.dd_cc5.value = self.cc5
            self.dd_cc5_ord.value = self.cc5_ord
        else:
            print('choose different caegory')

    def display_all(self):
        items_auto = [self.dd_cc5,self.dd_cc5_out]
        items_select = [self.dd_cc5_ord,self.dd_cc5_ord_out]
        items_0 = [self.button_next, self.output_next,self.button_save, self.output_save]
        box_auto = Box(children=items_auto, layout=Layout(display='flex',flex_flow='column', align_items='stretch', align_content='center', width='80%'))
        box_select = Box(children=items_select, layout=Layout(display='flex',flex_flow='column', align_items='stretch', align_content='center', width='80%'))
        box_0 = Box(children=items_0, layout=Layout(display='flex',flex_flow='row', align_items='stretch', align_content='center', width='80%'))
        display(VBox([box_select,self.text_widget1,self.text_widget2,self.text_widget3,box_auto, box_0]))
      
    def output_labels(self):
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)
        return self.df

    def start_to_label(self):
        self.init_widget()
        self.display_all()

        def on_value5_change(change):
            with self.dd_cc5_out:
                self.cc5 = change['new']
            self.dd_cc5_out.clear_output()

        def on_value5_ord_change(change):
            
            with self.dd_cc5_ord_out:
                self.cc5_ord = change['new']
            if self.use_probabilities:
                self.order()
            else:
                self.order2()
            self.pick_obs()
            #self.dd_cc5_ord_out.clear_output()
              
        def on_button_save_clicked(b):
            with self.output_save:
                self.df['cc3'].loc[self.idx] = coicop_5_3[self.cc5]
                self.df['cc4'].loc[self.idx] = coicop_5_4[self.cc5]
                self.df['cc5'].loc[self.idx] = self.cc5
                self.df['labeled_by'].loc[self.idx] = self.labeled_by
            self.counter += 1
            self.pick_obs()

        def on_button_next_clicked(b):
            self.counter += 1
            self.pick_obs()
        #self.dd_cc5_ord.observe(on_value5_ord_change, names='cat')
        self.dd_cc5.observe(on_value5_change, names='value')
        self.dd_cc5_ord.observe(on_value5_ord_change, names='value')
        self.button_save.on_click(on_button_save_clicked)
        self.button_next.on_click(on_button_next_clicked)
