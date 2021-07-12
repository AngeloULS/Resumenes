import torch

from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
from transformers import BertTokenizerFast, EncoderDecoderModel

app = Flask(__name__)
app.secret_key = 'mysecretkey'

usuario =[
    {
        "Nombre":"Maycol",
        "Apellido":"Gonzalez"
    },
    {
        "Nombre":"Pepe",
        "Apellido":"Rojas"
    }
]

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
        
    

@app.route('/usuario', methods=["GET"])
def getUsuario():
    return jsonify(usuario)


@app.route('/mayuscula/<oracion>', methods=["GET"])
def mayuscula(oracion):
    resultado =''
    for i in oracion.split():
        resultado += i[0].upper() + i[1:-1] + i[-1].upper() + ' '
        
    return resultado


@app.route('/nombre/<string:nom>', methods=["GET"])
def nombre(nom):
    return nom


@app.route('/', methods=["GET","POST"])
def index():
    if request.method == 'GET':
        if request.form.get('action1') == 'VALUE1':
            print('btn 1')
        
    return render_template('index.html')


@app.route("/f", methods=['GET', 'POST'])
def indexx():
    if request.method == 'POST':
        if request.form.get('action1') == 'VALUE1':
            print('btn 1')
            texto = request.form['input']
            print(texto)

            resultado =''
            resultado = resultado + modificaTexto(texto)
            
            flash(resultado)
        elif  request.form.get('action2') == 'VALUE2':
            print('btn 2')
        else:
            pass # unknown
    elif request.method == 'GET':
        return render_template('index.html')
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug= True)