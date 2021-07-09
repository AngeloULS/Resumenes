import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('mrm8488/bert2bert_shared-spanish-finetuned-muchocine-review-summarization')
tokenizer = T5Tokenizer.from_pretrained('mrm8488/bert2bert_shared-spanish-finetuned-muchocine-review-summarization')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

text ="""
La Guardia Civil ha desarticulado un grupo organizado dedicado a copiar en los examenes teoricos para la obtencion del permiso de conducir. Para ello, empleaban receptores y camaras de alta tecnologia y operaban desde la misma sede del Centro de examenes de la Direccion General de Trafico (DGT) en Mostoles. Es lo que han llamado la Operacion pinga. El grupo desarticulado ofrecia el servicio de transporte y tecnologia para copiar y poder aprobar. Por dicho servicio cobraban 1.000 euros. Los investigadores sorprendieron in fraganti a una mujer intentando copiar en el examen. Portaba una chaqueta con dispositivos electronicos ocultos, concretamente un telefono movil al que estaba conectada una camara que habia sido insertada en la parte frontal de la chaqueta para transmitir online el examen y que orientada al ordenador del Centro de Examenes en el que aparecen las preguntas, permitia visualizar las imagenes en otro ordenador alojado en el interior de un vehiculo estacionado en las inmediaciones del centro. En este vehiculo, se encontraban el resto del grupo desarticulado con varios ordenadores portatiles y tablets abiertos y conectados a paginas de test de la DGT para consultar las respuestas. Estos, comunicaban con la mujer que estaba en el aula haciendo el examen a traves de un diminuto receptor bluetooth que portaba en el interior de su oido.  Luis de Lama, portavoz de la Guardia Civil de Trafico destaca que los ciudadanos, eran de origen chino, y copiaban en el examen utilizando la tecnologia facilitada por una organizacion. Destaca que, ademas de parte del fraude que supone copiar en un examen muchos de estos ciudadanos desconocian el idioma, no hablan ni entienden el español lo que supone un grave riesgo para la seguridad vial por desconocer las señales y letreros que avisan en carretera de muchas incidencias. 
"""


preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)

# Summarized output from above ::::::::::
# the us has over 637,000 confirmed Covid-19 cases and over 30,826 deaths. 
# president Donald Trump predicts some states will reopen the country in april, he said. 
# "we'll be the comeback kids, all of us," the president says.