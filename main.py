from flask import Flask, render_template, request
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


app = Flask(__name__)

model = T5ForConditionalGeneration.from_pretrained('t5-small',force_download = True)
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')


def summarize(text,min_len,max_len):

    """
    # Function to Summarize text using the T5model
    """
    
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
    return render_template('index.html')

@app.route('/page1')
def page_1():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def data():

    """
        # ARTICLE is the text to be summarized
        # We inititialize min_len and max_len according to length of the chosen summary
        # We require that the length of ARTICLE be atleast 60 characters
    """
    
    if request.method == "POST":    
        ARTICLE = request.form['text'] 

        if len(ARTICLE) < 60:
            return render_template('result.html', Output='PARAGRAPH TOO SMALL')

        l = request.form['length']
     
        if l=='short':
            min_len, max_len = 50, 100
        if l=='medium':
            min_len, max_len = 100, 150
        if l=='long':
            min_len, max_len = 150, 200

        str1 = summarize(ARTICLE,min_len,max_len)
      
        return render_template('result.html', Output=str1)

if __name__ == "__main__":

    """
    # Pretrained T5 model for summarizing text

    """
    
    app.run(debug=False)