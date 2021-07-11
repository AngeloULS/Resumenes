import torch
from transformers import BertTokenizerFast, EncoderDecoderModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
tokenizer = BertTokenizerFast.from_pretrained(ckpt)
model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

def generate_summary(text):

   inputs = tokenizer([text], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
   input_ids = inputs.input_ids.to(device)
   attention_mask = inputs.attention_mask.to(device)
   output = model.generate(input_ids, attention_mask=attention_mask)
   return tokenizer.decode(output[0], skip_special_tokens=True)

sol = "El sistema solar se formó hace 4568 millones de años por el colapso gravitatorio de una parte de una nube molecular gigante. Esta nube primigenia tenía varios años luz de diámetro y probablemente dio a luz a varias estrellas.​ Como es normal en las nubes moleculares, consistía principalmente de hidrógeno, algo de helio y pequeñas cantidades de elementos pesados surgidos de previas generaciones estelares. A medida que la región —conocida como nebulosa protosolar—45​ se convertía en el sistema solar, colapsaba y la conservación del momento angular hizo que rotase más deprisa. El centro, donde se acumuló la mayor parte de la masa, se volvió cada vez más caliente que el disco circundante.44​ A medida que la nebulosa en contracción rotaba más deprisa, comenzó a aplanarse en un disco protoplanetario con un diámetro de alrededor de 200 UA44​ y una densa y caliente protoestrella en el centro.46​47​ Los planetas se formaron por acreción a partir de este disco48​ en el que el gas y el polvo atraídos gravitatoriamente entre sí se unen para formar cuerpos cada vez más grandes. En este escenario, cientos de protoplanetas podrían haber surgido en el temprano sistema solar que acabaron fusionándose o fueron destruidos dejando los planetas, los planetas enanos y el resto de cuerpos menores. Gracias a sus puntos de ebullición más altos, solo los metales y silicatos podían existir en forma sólida cerca del Sol, en el cálido sistema solar interior; estos fueron finalmente los componentes de Mercurio, Venus, la Tierra y Marte: los planetas rocosos. Debido a que los metales solo eran una pequeña parte de la nebulosa solar, los planetas terrestres no se podían hacer muy grandes. Los planetas gigantes (Júpiter, Saturno, Urano y Neptuno) se formaron más lejos, más allá de la línea de congelación: el límite entre las órbitas de Marte y Júpiter donde las temperaturas son lo suficientemente bajas como para que los compuestos volátiles permanezcan sólidos. Los hielos que forman estos planetas eran más abundantes que los metales y silicatos que formaron los planetas terrestres interiores, por lo que los permitió crecer hasta ser lo suficientemente masivos como para capturar grandes atmósferas de hidrógeno y helio: los elementos más ligeros y abundantes. Los residuos restantes que no llegaron a convertirse en planetas se agruparon en regiones como el cinturón de asteroides, el cinturón de Kuiper y la nube de Oort. El modelo de Niza explica la aparición de estas regiones y propone que los planetas exteriores se podrían haber formado en sitios diferentes de los actuales a los que habrían llegado tras múltiples interacciones gravitatorias. "

text = "Al filo de las 22.00 horas del jueves, la Asamblea de Madrid vive un momento sorprendente: Vox decide no apoyar una propuesta del PP en favor del blindaje fiscal de la Comunidad. Se ha roto la unidad de los tres partidos de derechas. Es un hecho excepcional. Desde que arrancó la legislatura, PP, Cs y Vox han votado en bloque casi el 75 de las veces en el pleno de la Cámara. Juntos decidieron la composición de la Mesa de la Asamblea. Juntos invistieron presidenta a Isabel Díaz Ayuso. Y juntos han votado la mayoría de proposiciones no de ley, incluida la que ha marcado el esprint final de la campaña para las elecciones generales: acaban de instar al Gobierno de España a la ilegalización inmediata de los partidos separatistas que atenten contra la unidad de la Nación. Los críticos de Cs no comparten el apoyo al texto de Vox contra el secesionisimo Ese balance retrata una necesidad antes que una complicidad, según fuentes del PP con predicamento en la dirección regional y nacional. Tras casi 15 años gobernando con mayoría absoluta, la formación conservadora vivió como una tortura la pasada legislatura, en la que dependió de Cs para sacar adelante sus iniciativas. El problema se agudizó tras las elecciones autonómicas de mayo. El PP ha tenido que formar con Cs el primer gobierno de coalición de la historia de la región, y ni siquiera con eso le basta para ganar las votaciones de la Cámara. Los dos socios gubernamentales necesitan a Vox, la menos predecible de las tres formaciones. Tenemos que trabajar juntos defendiendo la unidad del país, por eso no quisimos dejar a Vox solo, dijo ayer Díaz Ayuso para justificar el apoyo de PP y Cs a la proposición de la extrema derecha sobre Cataluña. Después nosotros llevábamos otra proposición para defender el blindaje fiscal de Madrid, y ahí Vox nos dejó atrás. No permitió que esto saliera. Es un grave error por su parte, prosiguió, recalcando el enfado del PP. Demuestra que está más en cuestiones electoralistas, subrayó. Los que pensamos, con nuestras inmensas diferencias, que tenemos cosas en común que nos unen como partidos que queremos Comunidades libres, con bajos impuestos, en las que se viva con seguridad y en paz, tenemos que estar unidos, argumentó. Y por lo menos nosotros de nuestra línea no nos separamos. Al contrario de lo que está ocurriendo el Ayuntamiento de Madrid, donde el PP y Cs ya han defendido posiciones de voto distintas, pese a compartir el Gobierno, en la Asamblea los partidos de Díaz Ayuso e Ignacio Aguado están actuando con la máxima lealtad en las votaciones del pleno. Otra cosa son las comisiones. Y el caso Avalmadrid. Es en ese terreno donde Cs y Vox están buscando el margen de maniobra necesario para separarse del PP en plena campaña electoral, abandonando a su suerte a su socio para distinguirse ante los electores. —Usted me ha dejado tirada, le espetó la presidenta de la Comunidad de Madrid a Rocío Monasterio tras saber que Vox permitiría que la izquierda tuviera mayoría en la comisión parlamentaria que investigará los avales concedidos por la empresa semipública entre 2007 y 2018, lo que podría incluir el de 400.000 euros aprobado en 2011, y nunca devuelto al completo, para una empresa participada por el padre de Isabel Díaz Ayuso. Monasterio no es de fiar. Dice una cosa y hace la contraria, dice una fuente popular sobre las negociaciones mantenidas para repartirse los puestos de las diferentes comisiones, que Vox no cumplió tras buscar un segundo pacto con otras formaciones (que no llegó a buen puerto). Ilegalización de Vox Los tres partidos de derechas también se han enfrentado por la ubicación de Vox en el pleno. Las largas negociaciones para la investidura de Díaz Ayuso dejaron heridas abiertas. Y los diputados de Cs no desaprovechan la oportunidad de lanzar dardos contra los de Vox, pero luego coinciden con ellos en la mayoría de votaciones. Ocurrió, por ejemplo, el jueves, cuando se debatía la polémica proposición para instar al Gobierno nacional a ilegalizar a los partidos separatistas que atenten contra la unidad de España. —Mostrar nuestra sorpresa ante la presentación por parte de Vox de esta propuesta, lanzó Araceli Gómez, diputada de la formación de Aguado. Sorprende que planteen ustedes este asunto cuando está también sobre la mesa el debate de su propia ilegalización por atentar contra el ordenamiento jurídico o contra valores constitucionales como la igualdad o la no discriminación. Luego de esa descalificación, y ante la incredulidad de los diputados de los partidos de izquierdas, Cs unió sus votos a los de Vox y a los del PP. La decisión ha provocado polémica interna, como demuestra que Albert Rivera no la apoyara ayer explícitamente. Tampoco ha sido bien acogida por el sector crítico de la formación. Pero ha demostrado una cosa: en Madrid hay tres partidos que casi siempre votan como uno."

noticia = """
Mañana 65 mesas de votación en San Ramón —que corresponde a 4 locales completos y otras dos mesas de otros recintos, un 25% de las mesas totales de la comuna— deberán repetir la votación de alcalde y concejales, luego de la resolución del Tricel, que acogió un requerimiento realizado por otros candidatos, por denuncias de irregularidades, como el exceso de votos asistidos, la falta de sellos para los votos, la incorporación de vocales de mesa voluntarios a las 9 horas, entre otras acusaciones que inculpaban a la candidatura del cuestionado Miguel Ángel Aguilera, ex PS, quien logró imponerse el 15 y 16 de mayo.

En este segundo proceso, la papeleta para alcalde tendrá los nombres de Aguilera, Gustavo Toro (DC), Miguel Bustamante (FA), Genaro Balladares (independiente), Miguel Pino (independiente) y David Cabedo (RN).
Entre las disposiciones, el Servel enfatizó que “cada vez que se utilice el voto asistido en una elección, el secretario de la mesa dejará constancia en acta de la existencia del sufragio asistido y de la identidad del sufragante y su asistente. No puede una misma persona asistir a más de un elector, a menos que sea un familiar directo”.
Optimismo y más confianza de candidatos

Entre los candidatos coincidieron en el llamado a votar, también en que existen mayores garantías del Servel y de sus propios equipos para realizar los comicios, por lo que se manifestaron confiados en que será un proceso más transparente.

Gustavo Toro, la carta DC y de Unidad Constituyente, se manifestó optimista. “Estamos bastante optimistas para mañana, atendido que sentimos que el proceso se nos está garantizando, que las instituciones ya están funcionando dentro de la comuna. Servel ha llevado adelante una tremenda tarea y labor, también seguridad”, dijo.

Toro, quien lideró las acciones contra las irregularidades y las denuncias contra el alcalde Aguilera, mostró confianza en su opción. “Sentimos que mañana se pone término a una historia oscura de la comuna y se comienzan a ver las luces de esperanza donde brilla el sol dentro de nuestra comuna, donde vamos a abrazar el triunfo con la verdad y la justicia, que ha sido la característica principal de esta candidatura”, sostuvo.

Miguel Bustamante (FA), expresó que tiene buenas expectativas para el proceso. “Hoy día hay muchas más garantías que en la primera elección, porque el Servel cambió a todos los vocales de mesa, cambió a los encargados de locales - porque había muchas denuncias que habían encargados de local ligados a la municipalidad - además entiendo que va a estar el director del Servel, va a estar la disposición temprana del material electoral, de las actas de escrutinio. Esperamos también que haya habido una mejor capacitación de los vocales de mesa, particularmente del protocolo del voto asistido”, opinó.

Pero además instó a poner atención a lo que sucede fuera de los locales de votación. “Durante el periodo especial de campaña que se abrió he conversado con muchos vecinos y he recibido varias denuncias de cohecho, de que gente ligada a Aguilera habría estado cerca de los locales de votación pagando votos , acarreando gente, entonces también tenemos que poner mucha atención fuera de los locales”, manifestó. Agregó que representa a “una fuerza joven, una fuerza nueva, alejada de los partidos políticos que le dieron sustento al aguilerismo durante 30 años”.

Mientras, David Cabedo (RN), instó a la participación en este proceso el día de mañana y se mostró confiado de los resultados. “Tengo la convicción de que mañana ganaremos, ya que los vecinos se inclinarán por la única opción de cambio real para nuestra comuna, ya que no hemos pactado como otros candidatos, los que han reconocido incluso en televisión sus pactos con los Olguín, con los Jaque y otros de la mafia de Aguilera”, arremetió.
"""

noticia2 = """
Fue el fiscal regional de La Araucanía, Miguel Rojas Thiele, quien confirmó que la persona fallecida en Carahue fue identificada como Pablo Marchant, tal como informó durante la mañana el líder de la CAM Héctor Llaitul.

"En horas de la madrugada se pudo determinar que se trata de don Pablo Marchant Gutiérrez quedando descartada aquella información que daba cuenta que la persona fallecida se trataba del hijo de don Héctor Llaitul, información que no fue emanada por parte de la fiscalía y que surgió a través de distintos medios de comunicación y de distintas redes sociales", aseguró el fiscal.

Según detalló el persecutor, entre las primeras diligencias, se permitió a Héctor Llaitul identificar el cuerpo de la persona fallecida, descartando que se tratara de su hijo Ernesto, lo que fue ratificado con análisis periciales que confirmaron la identidad de la persona muerta.

"De acuerdo a los primeros antecedentes recopilados, habrían participado alrededor de 6 a 7 personas, quienes se encontraban encapuchadas, vestían ropas oscuras y además portaban armas de grueso calibre, con las cuales atacan a funcionarios policiales de Carabineros que en ese momento se encontraban realizando labores de resguardo del predio donde acaecen estos hechos. Producto de este ataque, funcionarios policiales de Carabineros hacen uso de sus armas de servicio, generándose un enfrentamiento, producto del cual resulta fallecido uno de los presuntos atacantes, quien vestía ropas oscuras y además portaba un fusil M16", agregó el fiscal.

Según la fiscalía, las indagatorias siguen en curso, tanto para esclarecer el enfrentamiento, como también la muerte de Pablo Marchant y las heridas del trabajador, que "en estos momentos se encuentra conectado a un respirador mecánico".

"""
print(generate_summary(noticia2))