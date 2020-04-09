
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

with open(path+'/data/label_cc_dict.json') as json_file:#
    label_cc_dict = json.load(json_file)
    
class labeler:
    '''
    This object takes a dataframe of web scraped product data as input and displays the prediction of unlabeld data in a dropdown. 
    The user can accept or edit the prediction and save the label.
    '''
    def __init__(self,labeled_by, df,text_orig_str='text_orig',text_trans_str='text_trans',url_str='url', class_dict=label_cc_dict):
        self.counter = 0
        self.labeled_by = labeled_by
        self.text_orig_str = text_orig_str
        self.text_trans_str = text_trans_str
        self.url_str = url_str
        self.df = df.reset_index(drop=True) 
        self.df = df.sort_values('cc5_pred')
        if 'labeled_by' not in df.columns:
            self.df['labeled_by'] = None
        self.labels3 = class_dict['cc3']        
        self.labels4 = class_dict['cc4']
        self.labels5 = class_dict['cc5'] 
        self.labels3.append('99_Non-Food')       
        self.labels4.append('999_Non-Food')
        self.labels5.append('9999_Non-Food')
        self.labels3.sort()
        self.labels4.sort()
        self.labels5.sort()
        if 'cc3_labeled' in df.columns:
            self.df_idx =  list(self.df.index[self.df['cc3_labeled'].isna()])
        else:
            self.df['cc3_labeled'] = None
            self.df['cc4_labeled'] = None
            self.df['cc5_labeled'] = None
            self.df_idx =  list(range(0,len(self.df)))
        #random.shuffle(self.df_idx)
        self.idx =  self.df_idx[self.counter]
        try:
            self.text_orig = self.df[self.text_orig_str].loc[self.idx]
            self.text_trans = self.df[self.text_trans_str].loc[self.idx]
            self.cc3 = self.df['cc3_pred'].loc[self.idx]
            self.cc4 = self.df['cc4_pred'].loc[self.idx]
            self.cc5 = self.df['cc5_pred'].loc[self.idx]
            self.link = self.df[self.url_str].loc[self.idx]
        except:
            sys.exit('specify column names / dataframe needs columns: "text_orig","text_trans","cc3_pred","cc4_pred","cc5_pred","url"')

            
        self.new_labels = {}
        
    def init_widget(self):
        self.text_widget1 = widgets.Text(value=self.text_orig, description='Source text:',layout=Layout(width='80%', height='40px'), disabled=False)
        self.text_widget2 = widgets.Text(value=self.text_trans, description='Translated text:',layout=Layout(width='80%', height='40px'), disabled=False)
        self.dd_cc3 = widgets.Dropdown(options=self.labels3, value=self.cc3, description='COICOP 3:',layout=Layout(width='41%', height='30px'),  disabled=False) #, continuous_update=True
        self.dd_cc3_out = widgets.Output()
        self.dd_cc4 = widgets.Dropdown(options=self.labels4, value=self.cc4, description='COICOP 4:',layout=Layout(width='41%', height='30px'),  disabled=False)
        self.dd_cc4_out = widgets.Output()
        self.dd_cc5 = widgets.Dropdown(options=self.labels5, value=self.cc5, description='COICOP 5:',layout=Layout(width='41%', height='30px'),  disabled=False)
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
        self.text_orig = self.df[self.text_orig_str].loc[self.idx]
        self.text_trans = self.df[self.text_trans_str].loc[self.idx]
        self.cc3 = self.df['cc3_pred'].loc[self.idx]
        self.cc4 = self.df['cc4_pred'].loc[self.idx]
        self.cc5 = self.df['cc5_pred'].loc[self.idx]
        self.link = self.df[self.url_str].loc[self.idx]
        self.text_widget1.value = self.text_orig
        self.text_widget2.value = self.text_trans
        self.dd_cc3.value = self.cc3
        self.dd_cc4.value = self.cc4
        self.dd_cc5.value = self.cc5


    def display_all(self):

        items_auto = [self.dd_cc3,self.dd_cc3_out,self.dd_cc4,self.dd_cc4_out,self.dd_cc5,self.dd_cc5_out]
        items_0 = [self.button_next, self.output_next,self.button_save, self.output_save,self.button_link, self.output_link]
        box_auto = Box(children=items_auto, layout=Layout(display='flex',flex_flow='column', align_items='stretch', align_content='center', width='80%'))
        box_0 = Box(children=items_0, layout=Layout(display='flex',flex_flow='row', align_items='stretch', align_content='center', width='80%'))
        display(VBox([self.text_widget1,self.text_widget2,box_auto, box_0]))
      
    def output_labels(self):

        return self.df

    def start_to_label(self):
        self.init_widget()
        self.display_all()

        def on_value3_change(change):
            with self.dd_cc3_out:
                self.cc3 = change['new']
            self.dd_cc3_out.clear_output()

        def on_value4_change(change):
            with self.dd_cc4_out:
                self.cc4 = change['new']
            self.dd_cc4_out.clear_output()

        def on_value5_change(change):
            with self.dd_cc5_out:
                self.cc5 = change['new']
            self.dd_cc5_out.clear_output()
              
        def on_button_save_clicked(b):
            with self.output_save:
                #print(self.cc3,self.cc4,self.cc5)
                #self.new_labels[self.idx] = [self.cc3,self.cc4,self.cc5]
                self.df['cc3_labeled'].loc[self.idx] = self.cc3
                self.df['cc4_labeled'].loc[self.idx] = self.cc4
                self.df['cc5_labeled'].loc[self.idx] = self.cc5
                self.df['labeled_by'].loc[self.idx] = self.labeled_by
            self.pick_obs()

        def on_button_next_clicked(b):
            self.pick_obs()

        def on_button_link_clicked(b):
            print(self.link)


        self.dd_cc3.observe(on_value3_change, names='value')
        self.dd_cc4.observe(on_value4_change, names='value')
        self.dd_cc5.observe(on_value5_change, names='value')
        self.button_save.on_click(on_button_save_clicked)
        self.button_next.on_click(on_button_next_clicked)
        self.button_link.on_click(on_button_link_clicked)
