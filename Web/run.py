import torch

from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
from transformers import BertTokenizerFast, EncoderDecoderModel

app = Flask(__name__)
app.secret_key = 'mysecretkey'

def modificaTexto(texto):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
    tokenizer = BertTokenizerFast.from_pretrained(ckpt)
    model = EncoderDecoderModel.from_pretrained(ckpt).to(device)
    inputs = tokenizer([texto], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    
    resultado = tokenizer.decode(output[0], skip_special_tokens=True)

    return resultado


@app.route('/', methods=["GET","POST"])
def index():
    if request.method == 'POST':
        return render_template('index.html')
    elif request.method == 'GET':
        return render_template('index.html')


@app.route("/resumen", methods=['GET', 'POST'])
def indexx():
    if request.method == 'POST':
        if request.form.get('action1') == 'VALUE1':
            print('btn 1')
            texto = request.form['input']
            print(texto)
            resultado =''
            resultado = resultado + modificaTexto(texto)
            flash(resultado)
            flash(texto)

            return render_template('response.html', noticia = texto, resumen = resultado)
     
    elif request.method == 'GET':
        return render_template('index.html') #index.html
    
    return render_template("response.html") #index.html

if __name__ == "__main__":
    app.run(debug= True)