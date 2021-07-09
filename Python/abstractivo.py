import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert2bert_shared-spanish-finetuned-muchocine-review-summarization")

model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/bert2bert_shared-spanish-finetuned-muchocine-review-summarization")

sequence = ("El sistema solar​ es el sistema planetario que liga gravitacionalmente a un conjunto de objetos astronómicos que giran directa o " 
"indirectamente en una órbita alrededor de una única estrella conocida con el nombre de Sol."
" "
"La estrella concentra el 99,86 de la masa del sistema solar,​ y la mayor parte de la masa restante se concentra en ocho planetas cuyas "
"órbitas son prácticamente circulares y transitan dentro de un disco casi llano llamado plano eclíptico.6​ Los cuatro planetas más cercanos, " 
"considerablemente más pequeños, Mercurio, Venus, Tierra y Marte, también conocidos como los planetas terrestres, están compuestos "
"principalmente por roca y metal.7​8​ Mientras que los cuatro más alejados, denominados gigantes gaseosos o «planetas jovianos», más masivos "
"que los terrestres, están compuestos de hielo y gases. Los dos más grandes, Júpiter y Saturno, están compuestos principalmente de helio "
"e hidrógeno. Urano y Neptuno, denominados gigantes helados, están formados mayoritariamente por agua congelada, amoniaco y metano."
" "
"El Sol es el único cuerpo celeste del sistema solar que emite luz propia,1 debido a la fusión termonuclear del hidrógeno y su "
"transformación en helio en su núcleo.11​ El sistema solar se formó hace unos 4600 millones de años12​13​14​ a partir del colapso de una nube "
"molecular. El material residual originó un disco circunestelar protoplanetario en el que ocurrieron los procesos físicos que llevaron a la "
"formación de los planetas.​ El sistema solar se ubica en la actualidad en la nube Interestelar Local que se halla en la Burbuja Local del "
"brazo de Orión, de la galaxia espiral Vía Láctea, a unos 28 000 años luz del centro de esta. "
"Concepción artística del sistema solar y las órbitas de sus planetas"
" "
"El sistema solar es también el hogar de varias regiones compuestas por objetos pequeños. El cinturón de asteroides, ubicado entre Marte y "
"Júpiter, es similar a los planetas terrestres ya que está constituido principalmente por roca y metal. En este cinturón se encuentra el "
"planeta enano Ceres. Más allá de la órbita de Neptuno están el cinturón de Kuiper, el disco disperso y la nube de Oort, que incluyen "
"objetos transneptunianos formados por agua, amoníaco y metano principalmente. En este lugar existen cuatro planetas enanos: Haumea, "
"Makemake, Eris y Plutón, el cual fue considerado el noveno planeta del sistema solar hasta 2006. Este tipo de cuerpos celestes ubicados "
"más allá de la órbita de Neptuno son también llamados plutoides, los cuales junto a Ceres, poseen el suficiente tamaño para que se hayan "
"redondeado por efectos de su gravedad, pero que se diferencian principalmente de los planetas porque no han vaciado su órbita de cuerpos "
"vecinos. "
" "
"Adicionalmente a los miles de objetos pequeños de estas dos zonas, algunas docenas de los cuales son candidatos a planetas enanos, "
"existen otros grupos como cometas, centauros y polvo cósmico que viajan libremente entre regiones. Seis planetas y cuatro planetas "
"enanos poseen satélites naturales. El viento solar, un flujo de plasma del Sol, crea una burbuja de viento estelar en el medio "
"interestelar conocido como heliosfera, la que se extiende hasta el borde del disco disperso. La nube de Oort, la cual se cree que es la "
"fuente de los cometas de período largo, es el límite del sistema solar y su borde está ubicado a un año luz desde el Sol. "
" "
"A principios del año 2016 se publicó un estudio según el cual puede existir un noveno planeta en el sistema solar, al que dieron el "
"nombre provisional de Phattie.​ Se estima que el tamaño de Phattie sería entre el de Neptuno y la Tierra y que el hipotético planeta "
"sería de composición gaseosa.")

inputs = tokenizer.encode("summarize: " + sequence, return_tensors='pt', max_length=512, truncation=True)
outputs = model.generate(inputs, max_length=150, min_length=120, length_penalty=5, num_beams=2)
summary = tokenizer.decode(outputs[0])
print(summary)