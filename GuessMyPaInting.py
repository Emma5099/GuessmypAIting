import streamlit as st
import joblib
import pandas as pd
from io import StringIO 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
import random
import pathlib
from pathlib import Path
import shutil
from numpy.random import seed
import altair as alt
from vega_datasets import data
from tensorflow.keras.models import load_model
from PIL import Image
import io



#Fonction de troncation pour affichage pourcentage
def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

#Titre
st.title("Art Classification")
#

#Boutons
image_file = st.file_uploader("Choisissez un fichier ",accept_multiple_files=False)
model = st.selectbox("Model à utiliser",["CNN", "VGG16", "ResNet"])
prediction_a_faire = st.selectbox("Que voulez vous prédire",["Artiste", "Genre"])
#

#Différents modèles à charger
if (model=="CNN" and prediction_a_faire=="Artiste" ): 
    model=load_model("C:/Users/Emma/Documents/CPE/5A/Projet_Majeure/Projet/Models/Model_CNN_Artistes.h5")
if (model=="CNN" and prediction_a_faire=="Genre" ): 
        model=load_model('C:/Users/Emma/Documents/CPE/5A/Projet_Majeure/Projet/Models/Model_CNN_Genres.h5')
if(model=="VGG16" and prediction_a_faire=="Artiste" ):
    model=load_model("C:/Users/Emma/Documents/CPE/5A/Projet_Majeure/Projet/Models/model_VGG16_Artistes.h5")
if(model=="VGG16" and prediction_a_faire=="Genre" ):
    st.write("Il n'y a pas de modèle VGG16 disponible pour détecter les genres")
if(model=="ResNet" and prediction_a_faire=="Artiste" ):
    model=load_model("C:/Users/Emma/Documents/CPE/5A/Projet_Majeure/Projet/Models/model_ResNet_Artistes.h5")
if(model=="ResNet" and prediction_a_faire=="Genre" ):
    st.write("Il n'y a pas de modèle ResNet disponible pour détecter les genres")


#Lecture de l'image
if image_file is not None:
    # #Conversion de l'image uploadée en image normale qu'on sait manipuler
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels="BGR")
    opencv_image =cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    opencv_image = np.expand_dims(cv2.resize(opencv_image,(224,224)), axis=0) 


    #On recupère la liste des artistes/genre 
    liste_artistes=['Albrecht_Dürer', 'Alfred_Sisley', 'Edgar_Degas', 'Francisco_Goya', 'Marc_Chagall', 'Pablo_Picasso', 'Paul_Gauguin', 'Pierre-Auguste_Renoir', 'Rembrandt', 'Titian', 'Vincent_van_Gogh']
    liste_genre=['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism']
    #Dans le futur: faire la liste de TOUS artistes et des genres directement ici 


    if st.button('Predict'):
        #Détermination de la prediction (retourne liste de valeur pour chaque artistes/genres)
        Sortie_prediction=model.predict(opencv_image) 
        print(Sortie_prediction)

        
        #On récupère l'indice de la plus haute valeur (artiste/genre le plus probable)
        Premiere_prediction=np.argmax(Sortie_prediction, axis=-1) 
        Premiere_prediction=Premiere_prediction[0]
        print('Premier prediction : ' + str(Premiere_prediction))


        if (prediction_a_faire=="Artiste"):
            #Affichage de l'artiste/genre le plus probable
            nom=liste_artistes[Premiere_prediction].replace('_',' ')
            prediction1=truncate(Sortie_prediction[0][Premiere_prediction],3)*100
            st.write(" L'artiste deviné est " +str(nom)+ " avec une prediction de " + str(prediction1) + " % ",unsafe_allow_html=True)
            print('Premier prediction : ' + str(Premiere_prediction))



            #Affichage des probabilités selon les artsites/genres 
            st.write("Vous pouvez constatez les "+ str(prediction_a_faire) + " les plus probables ci-dessous:")

            print('type Sortie_prediction :' + str(type(Sortie_prediction)))
            print('type Artistes :' + str(type(liste_artistes)))
            print('type Sortie_prediction :' + str(len(Sortie_prediction)))
            print('type Artistes :' + str(len(liste_artistes)))
            data = pd.DataFrame({
                'Prediction': Sortie_prediction[0],
                'Artistes': liste_artistes,
            })

            cht=alt.Chart(data).mark_bar().encode(
                x='Prediction:Q',
                y=alt.Y('Artistes:N', sort='-x')
            )

            cht_expander = st.expander(str(prediction_a_faire) + 's les plus probables', expanded=True)
            cht_expander.altair_chart(cht, use_container_width=True)


        if (prediction_a_faire=="Genre"):
            nom=liste_genre[Premiere_prediction].replace('_',' ')
            prediction1=truncate(Sortie_prediction[0][Premiere_prediction],3)*100
            st.write(" Le genre deviné est " +str(nom)+ " avec une prediction de " + str(prediction1) + " % ",unsafe_allow_html=True)
            print('Premier prediction : ' + str(Premiere_prediction))



            #Affichage des probabilités selon les artsites/genres 
            st.write("Vous pouvez constatez les "+ str(prediction_a_faire) + " les plus probables ci-dessous:")

            print('type Sortie_prediction :' + str(type(Sortie_prediction)))
            print('type Artistes :' + str(type(liste_artistes)))
            print('type Sortie_prediction :' + str(len(Sortie_prediction)))
            print('type Artistes :' + str(len(liste_artistes)))
            data = pd.DataFrame({
                'Prediction': Sortie_prediction[0],
                'Artistes': liste_genre,
            })

            cht=alt.Chart(data).mark_bar().encode(
                x='Prediction:Q',
                y=alt.Y('Artistes:N', sort='-x')
            )

            cht_expander = st.expander(str(prediction_a_faire) + 's les plus probables', expanded=True)
            cht_expander.altair_chart(cht, use_container_width=True)




## Code pour faire tourner le site

#    ->     streamlit run GuessMyPainting.py