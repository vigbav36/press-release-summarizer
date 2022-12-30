import io
import torch
from flask import Flask, render_template, request



import pickle
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


app = Flask(__name__)

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

def summarize(text,min_len,max_len):
    
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=min_len,
                                        max_length=max_len,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return str(output)

@app.route('/')
def home_page():
    return render_template('index.html', button_name="SUBMIT")

@app.route('/page1')
def page_1():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def data():
    if request.method == "POST":
        ARTICLE = request.form['text']
        l = request.form['length']
     
        if l=='short':
            min_len, max_len = 30, 100
        if l=='medium':
            min_len, max_len = 50, 100
        if l=='long':
            min_len, max_len = 100, 150

        str1 = summarize(ARTICLE,min_len,max_len)
      
        return render_template('result.html', Output=str1, button_name="REDO")




