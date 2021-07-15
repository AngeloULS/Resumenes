import pandas as pd
import json 

test = 'C:\\Users\\angel\\Documents\\ULS 2021\\IA\\Repositorio\\Resumenes\\Data\\es_val.jsonl'
#'C:\\Users\\angel\\Documents\\ULS 2021\\IA\\Repositorio\\Resumenes\\Data\\es_test.jsonl'
#'C:\\Users\\angel\\Documents\\ULS 2021\\IA\\Repositorio\\Resumenes\\Data\\es_train.jsonl'
#'C:\\Users\\angel\\Documents\\ULS 2021\\IA\\Repositorio\\Resumenes\\Data\\es_val.jsonl'


Atest = []

with open(test) as contenTest:
    for linea in contenTest:
        Atest.append(json.loads(linea))

with open(f'{test}.json', 'w') as contenido:
    contenido.write(json.dumps(Atest))
 