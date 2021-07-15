#Necesita instalar newspaper3k y nltk
from newspaper.api import languages
import nltk, newspaper, time, json, os, io, json
from newspaper import Article
nltk.download('punkt')

class Noticia:
    """ Crea objeto noticia """
    def __init__(self, noticia, resumen):
        self.nombre = noticia
        self.cuerpo = resumen
        

def extraeNoticias(url, language="es"):
    noticia = Article(url)
    noticia.download()
    noticia.parse()
    noticia.download('punkt')
    
    aux = Noticia(noticia.text, noticia.summary)
    #article.nlp()
    #print("\n")
    #print("Titulo: "+ noticia.title)
    #print("\n")
    #print("Autor: "+ str(noticia.authors))
    #print("\n")
    print("Cueropo: " + str(noticia.text))
    print("Resuemn: " + str(noticia.summary))
    return aux

def noticiasJson(url):
    # También es válido 'C:\\Pruebas' o r'C:\Pruebas'
    dir = 'C:\\Users\\angel\\Documents\\ULS 2021\\IA\\Repositorio\\Resumenes\\Scraping\\Pruebas'
    file_name = "C:\\Users\\angel\\Documents\\ULS 2021\\IA\\Repositorio\\Resumenes\\Scraping\\Pruebas\\noticias.json"

    news = newspaper.build(url, languages="es")
    noticias = []

    for i in news.articles:
        print(extraeNoticias(i.url))
        noticias.append(extraeNoticias(i.url))
        time.sleep(10)
    print("\n")
    print("Listas: "+ str(noticias))

    datosJson=[json.dumps(s.__dict__, ensure_ascii=False) for s in noticias]
    with io.open(os.path.join(dir, file_name), 'w', encoding="utf-8") as file:
        json.dump(datosJson, file, ensure_ascii=False)


def variasNoticias(url):
    noticias = newspaper.build(url, languages="es")
    for noticia in noticias.articles:
        print("Noticia: ")
        print(noticia.url)
        extraeNoticias(noticia.url)
        print("\n")
        time.sleep(20)
        #print("\n")


#extraeNoticias("https://elpais.com/sociedad/2021-07-10/america-latina-vacuna-por-completo-a-un-15-de-su-poblacion-mientras-las-variantes-avanzan.html")
#variasNoticias("https://hipertextual.com/analisis")

#imprimeNoticia("https://elpais.com/internacional/2021-07-11/el-yihadismo-se-cuela-en-niger-por-el-agujero-de-la-pobreza.html")

noticiasJson("http://www.lanacion.cl/category/nacional/")