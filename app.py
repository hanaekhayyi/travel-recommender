from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re

# Charger les objets sauvegardés
svm = joblib.load('model/emotion_detection_model.pkl')
TFIDF_train = joblib.load('model/tfidf.pkl')
df = pd.read_csv('data/villes_finales.csv',encoding='ISO-8859-1')
sentiments = {0: 'sad', 1: 'happy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'curious'}

# Fonction de prédiction
def sentiment_pred(text):
    TFIDF_text = TFIDF_train.transform([text])
    prediction = svm.predict(TFIDF_text)
    return sentiments[prediction[0]]

####################################################################################
#                                                                                  #
#          Partie de la recommendation a partir du sentiment predit                #
#                                                                                  #
####################################################################################


# Les poids des sentiments
weights = {
    "sad": {"espace_vert": 5, "plage": 1, "Historical": 0, "musee": 0, "populationType_Grande": 0,"populationType_Petite": 5,"populationType_Moyenne": 3, "romantique": 1},
    "happy": {"espace_vert": 3, "plage": 5, "Historical": 1, "musee": 2, "populationType_Grande": 3,"populationType_Petite":1,"populationType_Moyenne":2, "romantique": 2},
    "love": {"espace_vert": 2, "plage": 2, "Historical": 5, "musee": 4, "populationType_Grande": 3,"populationType_Petite":2,"populationType_Moyenne":2, "romantique": 5},
    "anger": {"espace_vert": 4, "plage": 1, "Historical": 1, "musee": 2, "populationType_Grande": 0,"populationType_Petite":4,"populationType_Moyenne":1, "romantique": 1},
    "fear": {"espace_vert": 3, "plage": 1, "Historical": 1, "musee": 1,  "populationType_Grande": 0,"populationType_Petite":4,"populationType_Moyenne":1, "romantique": 1},
    "surprise": {"espace_vert": 1, "plage": 3, "Historical": 4, "musee": 5,  "populationType_Grande": 2,"populationType_Petite":3,"populationType_Moyenne":4, "romantique": 3},
    "curious": {"espace_vert": 1, "plage": 2, "Historical": 5, "musee": 5,  "populationType_Grande": 3,"populationType_Petite":2,"populationType_Moyenne":4, "romantique": 2},
}
def calculate_score(row, sentiment, adjusted_weights, preferences):
    # Exemple de calcul du score : la logique dépend des colonnes et du modèle de poids
    score = 0
    for feature in adjusted_weights:
        if feature in row:
            score += row[feature] * adjusted_weights[feature]  # Appliquer le poids ajusté
    return score


def recommendation(df, sentiment):
    df['Score'] = df.apply(lambda row: calculate_score(row, sentiment, weights), axis=1)

    # Trier les villes par score
    recommended_cities = df.sort_values(by="Score", ascending=False)
    top_cities = recommended_cities[["city", "country","Historical","espace_vert","plage"]][:5]
    df = df.drop('Score', axis=1)
    
    return top_cities

def get_recommendations(sentiment):
    data = recommendation(df,sentiment)  # Filtrer les données par sentiment
    return data


####################################################################################
#                                                                                  #
#                           Partie Site web                                        #
#                                                                                  #
####################################################################################

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/sentiment", methods=["GET", "POST"])
def express():
    return render_template("express.html")

@app.route('/recommendation', methods=["POST"])
def recommend():
    global df
    text = request.form["text"]
    sentiment = sentiment_pred(text) 
    # Récupérer les préférences de l'utilisateur (degrés d'appréciation)
    preferences = {
        "plage": int(request.form.get("plage", 5)),
        "espace_vert": int(request.form.get("espace_vert", 5)),
        "romantique": int(request.form.get("romantique", 5)),
        "Historical": int(request.form.get("Historical", 5)) if request.form.get("Historical") is not None else 5,
        "populationType_Grande": int(request.form.get("populationType_Grande", 5)),
        "populationType_Petite": int(request.form.get("populationType_Petite", 5)),
        "populationType_Moyenne": int(request.form.get("populationType_Moyenne", 5)),
    }

    print(f"Plage: {preferences['plage']}, Espace Vert: {preferences['espace_vert']}, Romantique: {preferences['romantique']}")

    adjusted_weights = weights[sentiment].copy()  

    # Ajuster les poids en fonction des préférences de l'utilisateur
    for feature, preference in preferences.items():
        if feature in adjusted_weights:
            adjusted_weights[feature] *= preference / 5  

    
    df['Score'] = df.apply(lambda row: calculate_score(row, sentiment, adjusted_weights, preferences), axis=1)

    print(df[['city', 'Score']].head())  

    
    recommended_cities = df.sort_values(by="Score", ascending=False)
    top_cities = recommended_cities[["city", "country", "Historical", "espace_vert", "plage","musées"]][:6]
    df = df.drop('Score', axis=1)

    
    print("Villes recommandées:")
    for index, row in top_cities.iterrows():
        print(f"Ville: {row['city']}, Pays: {row['country']}, Historique: {row['Historical']}, "
            f"Espace Vert: {row['espace_vert']}, Plage: {row['plage']}")
        # Vérification de la structure de la liste des musées
        if isinstance(row['musées'], list):
            museums_str = ', '.join(row['musées'])  # Convertir la liste en une seule chaîne
        else:
            museums_str = str(row['musées'])  # Si ce n'est pas une liste, la convertir en chaîne
        
        # Affichage des musées
        print(f"Musées: {museums_str}")
        print("-" * 40)  # Ligne pour séparer les villes

    # Formater les musées pour chaque ville
    top_cities['musées'] = top_cities['musées'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else str(x)  # Convertir chaque liste en chaîne
    )
    top_cities['musées'] = top_cities['musées'].apply(
        lambda x: x.split(",") if isinstance(x, str) else x 
    )
    # Nettoyer la liste pour supprimer les espaces superflus autour des noms de musées
    top_cities['musées'] = top_cities['musées'].apply(
        lambda x: [i.strip() for i in x] if isinstance(x, list) else x  # Supprimer les espaces autour des noms
    )
    top_cities['musées'] = top_cities['musées'].apply(
        lambda x: [museum.strip('[],"\'') for museum in x] if isinstance(x, list) else x
    )


    has_more = len(top_cities) == 6  
    return render_template(
        "result.html",
        sentiment=sentiment,
        recommendations=top_cities,
        has_more=has_more,
    )



if __name__ == "__main__":
    app.run(debug=True)
