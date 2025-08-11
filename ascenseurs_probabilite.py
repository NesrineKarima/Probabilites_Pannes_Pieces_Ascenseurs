import pandas as pd
import numpy as np
from docplex.mp.model import Model
import time

# Étape 1 – Récupération et Préparation des données
# ----------------------------------
input_path = "historique_pannes_ascenseurs.xlsx"
df = pd.read_excel(input_path)

required_cols = [
    "Event_ID", "Date", "ID_Ascenseur", "Pièce", "Nb_pannes",
    "Durée de vie (heures)", "Heures depuis dernier changement", "Adresse"
]
missing = set(required_cols) - set(df.columns)
if missing:
    raise KeyError(f"\033[1;31mColonnes manquantes : {missing}\033[0m")

# Conversion et tri
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['ID_Ascenseur', 'Pièce', 'Date']).reset_index(drop=True)

# Étape 2 – Calcul du ratio d’usure
# ----------------------------------
df['ratio_usure'] = df['Heures depuis dernier changement'] / df['Durée de vie (heures)']

# Étape 3 – Normalisation des pannes récentes
# --------------------------------------------
nb_max = df['Nb_pannes'].max() or 1
df['nb_pannes_norm'] = df['Nb_pannes'] / nb_max

# Étape 4 – Normalisation de la mémoire historique
# -------------------------------------------------
df['cum_pannes'] = df.groupby(['ID_Ascenseur', 'Pièce'])['Nb_pannes'].cumsum() - df['Nb_pannes']
cum_max = df['cum_pannes'].max() or 1
df['mem_pannes_norm'] = df['cum_pannes'] / cum_max

# Étape 5 – Score global par pièce
# ---------------------------------
alpha, beta, gamma = 0.6, 0.2, 0.2
df['score'] = (
    alpha * df['ratio_usure'] +
    beta * df['nb_pannes_norm'] +
    gamma * df['mem_pannes_norm']
)

# Étape 6 – Sélection des derniers états
# ---------------------------------------
cols_for_latest = ['ID_Ascenseur', 'Pièce', 'Adresse', 'Nb_pannes',
                   'ratio_usure', 'cum_pannes', 'score']
df_latest = df.groupby(['ID_Ascenseur', 'Pièce'], as_index=False).last()[cols_for_latest]

# Étape intermédiaire – paramétrage de CPLEX 
# -------------------------------------------------------------------------
model = Model(name="dummy_probabilite")
x = model.continuous_var(name="x")
model.add_constraint(x <= 1)
model.set_objective("min", x)

print("\033[1;36m Infos sur le modèle CPLEX :\033[0m")
model.print_information()
start = time.time()
_ = model.solve(log_output=True)
elapsed = time.time() - start
print(f"\033[1;35m Temps de résolution (dummy) : {elapsed:.4f} secondes\033[0m")

# Étape 7 – Application de la fonction softmax
# ---------------------------------------------
def softmax(s):
    exps = np.exp(s)
    return exps / exps.sum()

# Calcul de la probabilité numérique (float)
df_latest['proba_num'] = (
    df_latest.groupby('Pièce')['score']
             .transform(lambda s: softmax(s))
)

# Détermination des bornes pour trois niveaux de risque
def compute_bounds(probas):
    min_p = probas.min()
    max_p = probas.max()
    delta = (max_p - min_p) / 3
    return min_p + delta, min_p + 2 * delta

borne1, borne2 = compute_bounds(df_latest['proba_num'])

# Nouvelle fonction de label basée sur la probabilité
def label_from_proba(p):
    if p < borne1:
        return "Risque faible"
    elif p < borne2:
        return "Risque moyen"
    else:
        return "Risque élevé"

# Application du label
df_latest['Explication'] = df_latest['proba_num'].apply(label_from_proba)

# Mise en forme de la probabilité pour l’export
df_latest['Probabilité de panne'] = (
    df_latest['proba_num']
           .apply(lambda x: f"{x:.4f}".replace('.', ','))
)

# Résultats finaux – Format final à écrire
# ----------------------------------------
output_df = df_latest[['ID_Ascenseur', 'Pièce', 'Adresse',
                       'Probabilité de panne', 'Explication']]

# Écriture du fichier de sortie
output_csv = "distribution_probabilites_par_piece.csv"
output_df.to_csv(output_csv, index=False, encoding='utf-8-sig', sep=';', decimal=',')

# Message final en console
print(
    f"\033[38;5;214m✅ Distribution de probabilité générée dans : "
    f"\033[93m{output_csv}\033[38;5;214m\033[0m"
)
