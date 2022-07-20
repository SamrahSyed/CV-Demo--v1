import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

def Abs_Sum(text):
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=200,
                                    early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    strlist=output.split('. ')
    newString=''

    for val in strlist:
        newString+=val.capitalize()+'. '
    return newString
    #print ("\n\nOriginal text: \n", text)
    #print ("\n\nSummarized text: \n\n", newString)

    
#text ="""
#Most Americans, encouraged by Biden himself, had already expected that kind of normality to be restored and may be in no mood to contemplate months more of deprivation. The spike in Covid-19 cases that has hit many areas of the country has already turned what was sold as a summer of freedom from the virus into a replay of some of the worst parts of the pandemic as hospitals throughout the South are overrun by Covid patients. And conservatives have already long ago turned against Fauci, one of the world's most respected public health experts, and he is a top target of right-wing media.The last 17 months that changed the daily fabric of American life have been far from predictable. And there is some data from abroad -- albeit in more vaccinated nations like Britain and Israel -- that suggests the current Delta variant wave of the virus could ease or may not produce the same level of deaths as earlier surges. If so, its political impact could be mitigated.But even the prospect that the end of the battle against Covid-19 could be many, many months away represents a nightmare political scenario for the President and his Democratic Party, already facing historic headwinds in trying to keep control of Congress. They will now face the possibility of having to do so in a nation even more exhausted by a crisis that has already cost more than 620,000 lives and that has become more politically divided by the virus every month it rages on.
#"""
#Abs_Sum (text) 

from gensim.summarization.summarizer import summarize
def Ext_Sum(text, ratio=.5):
    newString=''
    for val in summarize(text, ratio=ratio).split():
        newString+=val+' '
    return newString
"""     print ("\n\nOriginal text: \n", text)
    print("\n\nSummarized text: \n\n", newString)
text ="""
#Most Americans, encouraged by Biden himself, had already expected that kind of normality to be restored and may be in no mood to contemplate months more of deprivation. The spike in Covid-19 cases that has hit many areas of the country has already turned what was sold as a summer of freedom from the virus into a replay of some of the worst parts of the pandemic as hospitals throughout the South are overrun by Covid patients. And conservatives have already long ago turned against Fauci, one of the world's most respected public health experts, and he is a top target of right-wing media.The last 17 months that changed the daily fabric of American life have been far from predictable. And there is some data from abroad -- albeit in more vaccinated nations like Britain and Israel -- that suggests the current Delta variant wave of the virus could ease or may not produce the same level of deaths as earlier surges. If so, its political impact could be mitigated.But even the prospect that the end of the battle against Covid-19 could be many, many months away represents a nightmare political scenario for the President and his Democratic Party, already facing historic headwinds in trying to keep control of Congress. They will now face the possibility of having to do so in a nation even more exhausted by a crisis that has already cost more than 620,000 lives and that has become more politically divided by the virus every month it rages on.
"""
Ext_Sum(text=text, ratio=.5) """