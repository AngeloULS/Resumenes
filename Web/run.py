from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
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
    resultado =''
    for i in texto.split():
        resultado += i[0].upper() + i[1:-1] + i[-1].upper() + ' '
        
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


@app.route('/home', methods=["GET","POST"])
def index():
    if request.method == 'GET':
        if request.form.get('action1') == 'VALUE1':
            print('btn 1')
            texto = request.form['input']
            print(texto)
            resultado =''
            for i in texto.split():
                resultado += i[0].upper() + i[1:-1] + i[-1].upper() + ' '
            flash(resultado)
        
    return render_template('index.html')


@app.route("/f", methods=['GET', 'POST'])
def indexx():
    if request.method == 'POST':
        if request.form.get('action1') == 'VALUE1':
            print('btn 1')
            texto = request.form['input']
            print(texto)
            resultado =''
            for i in texto.split():
                resultado += i[0].upper() + i[1:-1] + i[-1].upper() + ' '
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