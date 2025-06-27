import os
import json
import dash
from dash import dcc, html, Input, Output, State, ALL, Dash
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import base64
import io
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import datetime

app = Dash(__name__)
server = app.server 

app.layout = html.Div("Hello, Railway!")



PATIENTS_DIR = "patients"
os.makedirs(PATIENTS_DIR, exist_ok=True)

# Identifiants autorisés (à adapter)
VALID_USERS = {"admin": "motdepasse123"}

def list_patients():
    # Retourne la liste des patients (fichiers .json)
    return [f[:-5] for f in os.listdir(PATIENTS_DIR) if f.endswith(".json")]

def save_patient(prenom, nom, infos, df):
    key = f"{prenom}_{nom}".replace(" ", "_")
    # Sauvegarde infos
    with open(os.path.join(PATIENTS_DIR, f"{key}.json"), "w", encoding="utf-8") as f:
        json.dump(infos, f)
    # Sauvegarde CSV
    df.to_csv(os.path.join(PATIENTS_DIR, f"{key}.csv"), index=False)
    # Création du dossier patient si nécessaire
    patient_dir = os.path.join(PATIENTS_DIR, key)
    os.makedirs(patient_dir, exist_ok=True)
    return key

def load_patient(key):
    # Charge infos et CSV
    with open(os.path.join(PATIENTS_DIR, f"{key}.json"), encoding="utf-8") as f:
        infos = json.load(f)
    df = pd.read_csv(os.path.join(PATIENTS_DIR, f"{key}.csv"))
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return infos, df

def save_six_min_test(patient_key, df):
    """Sauvegarde le test de 6 minutes pour un patient"""
    # Création du dossier si nécessaire
    patient_dir = os.path.join(PATIENTS_DIR, patient_key)
    os.makedirs(patient_dir, exist_ok=True)
    # Sauvegarde du test de 6 minutes
    df.to_csv(os.path.join(patient_dir, "six_min_test.csv"), index=False)

def load_six_min_test(patient_key):
    """Charge le test de 6 minutes d'un patient s'il existe"""
    six_min_test_path = os.path.join(PATIENTS_DIR, patient_key, "six_min_test.csv")
    if os.path.exists(six_min_test_path):
        df = pd.read_csv(six_min_test_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        return df
    return None

# Initialisation des structures de données vides
df = pd.DataFrame(columns=['DateTime', 'FileName', 'Speed', 'Length', 'Height', 'HourOfDay', 'WalkingMinutes'])
patient_info = {}
perf_data = pd.Series({'Speed': 0, 'Length': 0, 'Height': 0})
nb_jours = 0
nb_pas_total = 0
nb_pas_moyen = 0
distance_totale = 0
distance_moyenne = 0

metrics = [
    ("Vitesse de pas (m/s)", perf_data['Speed']),
    ("Longueur de pas (m)", perf_data['Length']),
    ("Hauteur de pas (m)", perf_data['Height']),
    ("Nombre de pas", nb_pas_moyen),
    ("Distance parcourue (m)", distance_moyenne)
]

# --- Préparation des segments pour le graphique de densité ---
segments = []

# --- Statistiques quotidiennes ---
daily_stats = {}
dates = []

# Variables globales pour stocker les données du test de 6 minutes
six_min_test_data = None

# --- Dash app ---
app = dash.Dash(__name__)
app.title = "Gait Analysis Dash"
app.config.suppress_callback_exceptions = True
#app._favicon = "🦵"


def indicator_bar(value, vmin, vmax, label, total=None):
    # Ajuster les échelles pour le test de 6 minutes
    # Pour nombre de pas et distance parcourue, adapter l'échelle si c'est ces métriques
    if label == "Nombre de pas" and total is not None:
        # Ajuster l'échelle pour le nombre de pas du 6MWT
        # Un test de 6 minutes donne généralement entre 500-1000 pas
        vmin_6mwt = 0
        vmax_6mwt = 1200  # Valeur maximale ajustée pour le test de 6 minutes
    elif label == "Distance parcourue (m)" and total is not None:
        # Ajuster l'échelle pour la distance du 6MWT
        # Un test de 6 minutes donne généralement 400-700m
        vmin_6mwt = 0
        vmax_6mwt = 800  # Valeur maximale ajustée pour le test de 6 minutes
    else:
        # Pour les autres métriques, utiliser la même échelle
        vmin_6mwt = vmin
        vmax_6mwt = vmax

    # Définir les bornes des zones
    zone1 = vmin + (vmax - vmin) / 3
    zone2 = vmin + 2 * (vmax - vmin) / 3
    percent = 100 * (value - vmin) / (vmax - vmin)
    percent = min(max(percent, 0), 100)
    
    # Calcul du pourcentage pour le test de 6 minutes (si disponible)
    six_min_percent = None
    if total is not None:
        six_min_percent = 100 * (total - vmin_6mwt) / (vmax_6mwt - vmin_6mwt)
        six_min_percent = min(max(six_min_percent, 0), 100)
    
    # Couleurs des zones et des barres
    background_color = "#eeeeee"  # Couleur gris très clair pour le fond de la barre
    zone_colors = ["#E7174A", "#FF9A16", "#2CC1AA"]  # Rouge, Orange, Vert
    zone_labels = ["Mauvais", "Bon", "Excellent"]
    six_min_color = "#4da7f8"  # Bleu clair pour les valeurs du 6MWT
    daily_color = "#555"  # Gris foncé pour les valeurs quotidiennes
    title_color = "#2CC1AA"  # Turquoise pour les titres
    
    # Déterminer la zone actuelle en fonction de la valeur
    current_zone = 0
    if value >= zone1:
        current_zone = 1
    if value >= zone2:
        current_zone = 2
    
    # Label de la zone actuelle
    zone_label = "GOOD"  # Par défaut
    
    # Formater les valeurs pour l'affichage
    value_formatted = f"{value:.2f}" if isinstance(value, float) else f"{int(value)}"
    
    # Label pour la valeur du test de 6 minutes (si disponible)
    six_min_label = None
    if total is not None:
        if isinstance(total, float):
            six_min_label = f"{total:.2f}"
        else:
            six_min_label = f"{int(total)}"
    
    # Créer le texte de la valeur (combiné)
    if six_min_label:
        display_text = html.Div([
            html.Span(value_formatted, style={"color": daily_color, "fontWeight": "bold"}),
            html.Span(" | ", style={"color": "#666", "margin": "0 0.3em"}),
            html.Span(six_min_label, style={"color": six_min_color, "fontWeight": "bold"})
        ], style={"fontSize": "1.8em", "display": "flex", "alignItems": "center"})
    else:
        display_text = html.Span(value_formatted, style={"fontWeight": "bold", "fontSize": "2em", "color": daily_color})
    
    return html.Div([
        html.Div([
            # Titre avec alignement à gauche
            html.Div([
                html.Span(f"{label}", style={"fontWeight": "bold", "fontSize": "1.1em", "color": title_color})
            ], style={"textAlign": "left", "marginBottom": "0.5em"}),
            
            # Valeurs avec alignement à gauche strict
            html.Div([
                display_text
            ], style={"textAlign": "left", "marginBottom": "2em"}),  # Augmenté l'espace ici
            
            # Barre avec les 3 zones colorées - déplacée vers le bas
            html.Div([
                # Fond de la barre (gris clair)
                html.Div(style={
                    "width": "100%",
                    "height": "10px",
                    "background": background_color,
                    "position": "relative",
                    "borderRadius": "5px",
                    "display": "flex",  # Utiliser flexbox pour les zones
                }),
                
                # Les trois zones colorées
                html.Div(style={
                    "width": "33.33%",
                    "height": "10px",
                    "background": zone_colors[0],  # Rouge
                    "borderTopLeftRadius": "5px",
                    "borderBottomLeftRadius": "5px",
                    "position": "absolute",
                    "top": "0",
                    "left": "0",
                    "zIndex": 2
                }),
                html.Div(style={
                    "width": "33.33%",
                    "height": "10px",
                    "background": zone_colors[1],  # Orange
                    "position": "absolute",
                    "top": "0",
                    "left": "33.33%",
                    "zIndex": 2
                }),
                html.Div(style={
                    "width": "33.34%",
                    "height": "10px",
                    "background": zone_colors[2],  # Vert
                    "borderTopRightRadius": "5px",
                    "borderBottomRightRadius": "5px",
                    "position": "absolute",
                    "top": "0",
                    "left": "66.66%",
                    "zIndex": 2
                }),
                
                # Curseur quotidien
                html.Div(style={
                    "width": "3px",  # Largeur du curseur
                    "height": "16px",  # Hauteur du curseur
                    "background": "#333333",  # Noir
                    "position": "absolute",
                    "top": "-3px",  # Légèrement plus haut
                    "left": f"{percent}%",
                    "transform": "translateX(-50%)",  # Centrer sur la position
                    "zIndex": 4,
                    "borderRadius": "1px",
                }),
                
                # Curseur 6MWT (si disponible)
                html.Div(style={
                    "width": "3px",  # Largeur du curseur
                    "height": "16px",  # Hauteur du curseur
                    "background": six_min_color,  # Bleu
                    "position": "absolute",
                    "top": "-3px",  # Légèrement plus haut
                    "left": f"{six_min_percent}%" if six_min_percent is not None else "0%",
                    "transform": "translateX(-50%)",  # Centrer sur la position
                    "zIndex": 3,
                    "borderRadius": "1px",
                    "display": "block" if six_min_percent is not None else "none"
                })
            ], style={
                "position": "relative",
                "width": "100%",
                "height": "10px",
                "marginBottom": "0.5em",
                "marginTop": "2em"  # Augmenté l'espace au-dessus de la barre
            }),
            
            # Label de zone (GOOD) en dessous, aligné à droite
            html.Div([
                html.Span(zone_label, style={
                    "color": zone_colors[2],  # Vert
                    "fontSize": "0.9em",
                    "fontWeight": "bold"
                })
            ], style={
                "width": "100%",
                "textAlign": "right",
                "marginTop": "0.2em"
            }),
            
            # Légende des indicateurs (Quotidien/6MWT) - déplacée vers le bas
            html.Div([
                html.Div([
                    html.Div(style={
                        "width": "8px", 
                        "height": "8px", 
                        "background": "#333333",  # Noir pour quotidien
                        "display": "inline-block", 
                        "marginRight": "5px", 
                        "borderRadius": "50%"  # Cercle au lieu de carré
                    }),
                    html.Span("Quotidien", style={"fontSize": "0.8em", "color": daily_color})
                ], style={"display": "inline-block", "marginRight": "15px"}),
                html.Div([
                    html.Div(style={
                        "width": "8px", 
                        "height": "8px", 
                        "background": six_min_color,  # Bleu pour 6MWT
                        "display": "inline-block", 
                        "marginRight": "5px", 
                        "borderRadius": "50%"  # Cercle au lieu de carré
                    }),
                    html.Span("6MWT", style={"fontSize": "0.8em", "color": six_min_color})
                ], style={
                    "display": "inline-block", 
                    "marginRight": "5px", 
                    "visibility": "visible" if six_min_percent is not None else "hidden"
                })
            ], style={"width": "100%", "textAlign": "right", "marginTop": "1.5em"})  # Augmenté l'espace
        ], style={"padding": "1.5em"})
    ], style={
        "background": "#ffffff",  # Fond blanc
        "borderRadius": "12px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
        "margin": "0.5em",
        "height": "220px",
        "width": "100%",
        "color": "#333333"  # Texte gris foncé
    })

def make_gauge_bar(value, title, vmin, vmax, steps, total=None):
    # steps: [(start, end, color, label), ...]
    bar_color = "black"
    fig = go.Figure()
    # Ajout des zones colorées
    for (start, end, color, label) in steps:
        fig.add_shape(
            type="rect",
            x0=start, x1=end, y0=0.25, y1=0.75,
            fillcolor=color, line=dict(width=0),
            layer="below"
        )
        # Label de zone
        fig.add_annotation(
            x=(start+end)/2, y=0.85, text=label, showarrow=False,
            font=dict(size=10, color=color), yanchor="bottom"
        )
    # Barre principale
    fig.add_trace(go.Scatter(
        x=[vmin, vmax], y=[0.5, 0.5],
        mode="lines", line=dict(color="#EEEEEE", width=16), showlegend=False
    ))
    # Curseur valeur
    fig.add_trace(go.Scatter(
        x=[value], y=[0.5],
        mode="markers", marker=dict(color=bar_color, size=24, symbol="line-ns-open"), showlegend=False
    ))
    # Valeur numérique
    fig.add_annotation(
        x=vmin, y=0.5, text=f"{title}", showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=13, color="black")
    )
    fig.add_annotation(
        x=vmin, y=0.5, text=f"{value:.3g}" if isinstance(value, float) else f"{int(value)}",
        showarrow=False, xanchor="left", yanchor="top", font=dict(size=22, color="black"), yshift=-18
    )
    # Total en sous-titre
    if total is not None:
        fig.add_annotation(
            x=vmin, y=0.2, text=f"Total: {int(total)}", showarrow=False,
            xanchor="left", font=dict(size=12, color="gray")
        )
    fig.update_layout(
        xaxis=dict(range=[vmin, vmax], showticklabels=False, showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, fixedrange=True),
        height=80, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Roboto, Arial, sans-serif", size=15)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def make_density_figure(df):
    # Cas où aucune donnée n'est disponible
    if len(df) == 0 or 'FileName' not in df.columns or df['FileName'].nunique() == 0:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(title="Heure de la journée", range=[6, 18]),
            yaxis=dict(title="Jour"),
            annotations=[dict(
                text="Aucune donnée disponible",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                font=dict(size=20)
            )],
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Roboto, Arial, sans-serif", size=15)
        )
        return fig
        
    # Mapping FileName -> date formatée
    file_name_to_date = {}
    for file_name in df['FileName'].unique():
        try:
            date_str = ''.join(filter(str.isdigit, file_name))[:8]
            formatted_date = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]}"
        except:
            formatted_date = file_name
        file_name_to_date[file_name] = formatted_date

    y_labels = [file_name_to_date[f] for f in sorted(df['FileName'].unique())]
    traces = []
    for i, file_name in enumerate(sorted(df['FileName'].unique())):
        segs = [s for s in segments if s["file_name"] == file_name]
        y_val = file_name_to_date[file_name]
        for seg in segs:
            color = f"hsl({i*40%360},70%,50%)"
            if seg["duration"] >= 2:
                # Segment cliquable
                traces.append(go.Scatter(
                    x=[seg["start_time"], seg["end_time"]],
                    y=[y_val, y_val],
                    mode="lines",
                    line=dict(width=15, color=color, shape="linear"),
                    hoverinfo="text",
                    text=f"{y_val}<br>Début: {seg['start_time']:.2f}h<br>Fin: {seg['end_time']:.2f}h<br>Durée: {seg['duration']:.1f} min",
                    customdata=[f"{file_name}|{seg['start_time']}|{seg['end_time']}"]*2,
                    showlegend=False
                ))
            else:
                # Segment non cliquable (pas de customdata)
                traces.append(go.Scatter(
                    x=[seg["start_time"], seg["end_time"]],
                    y=[y_val, y_val],
                    mode="lines",
                    line=dict(width=15, color=color, shape="linear", dash="dot"),
                    hoverinfo="text",
                    text=f"{y_val}<br>Début: {seg['start_time']:.2f}h<br>Fin: {seg['end_time']:.2f}h<br>Durée: {seg['duration']:.1f} min",
                    customdata=[None, None],
                    showlegend=False,
                    opacity=0.4
                ))
    fig = go.Figure(traces)
    fig.update_layout(
        xaxis=dict(title="Heure de la journée", range=[6, 18]),
        yaxis=dict(title="Jour", categoryorder="array", categoryarray=y_labels[::-1]),
        margin=dict(t=50, b=40),
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Roboto, Arial, sans-serif", size=15)
    )
    return fig

def make_detail_figure(segment_data):
    # Vérifier si les données sont vides
    if segment_data is None or len(segment_data) == 0:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(
                text="Aucune donnée disponible",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                font=dict(size=20)
            )],
            height=600,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Roboto, Arial, sans-serif", size=15)
        )
        return fig
        
    def smooth(y, window=15):
        return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean()
    
    # Calculer les moyennes et les statistiques
    vitesse_moyenne = segment_data['Speed'].mean()
    hauteur_moyenne = segment_data['Height'].mean()
    longueur_moyenne = segment_data['Length'].mean()
    
    # Récupérer les heures de début et de fin pour calculer la durée
    start_time = segment_data['HourOfDay'].iloc[0] if 'HourOfDay' in segment_data.columns else 0
    end_time = segment_data['HourOfDay'].iloc[-1] if 'HourOfDay' in segment_data.columns else 0
    duree_minutes = (end_time - start_time) * 60
    nb_pas = len(segment_data) * 2
    
    # Créer les sous-graphiques sans titres
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    
    # Vitesse
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=segment_data['Speed'],
        mode='markers', marker=dict(color='lightgray', size=6), name='Vitesse (brut)',
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=smooth(segment_data['Speed']),
        mode='lines', line=dict(color='blue', width=2), name='Vitesse (lissé)',
        showlegend=False
    ), row=1, col=1)
    # Ligne de moyenne pour la vitesse
    fig.add_trace(go.Scatter(
        x=[segment_data['DateTime'].iloc[0], segment_data['DateTime'].iloc[-1]],
        y=[vitesse_moyenne, vitesse_moyenne],
        mode='lines', line=dict(color='red', width=1.5, dash='dash'),
        name='Moyenne', showlegend=False
    ), row=1, col=1)
    
    # Hauteur
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=segment_data['Height'],
        mode='markers', marker=dict(color='lightgray', size=6), name='Hauteur (brut)',
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=smooth(segment_data['Height']),
        mode='lines', line=dict(color='blue', width=2), name='Hauteur (lissé)',
        showlegend=False
    ), row=2, col=1)
    # Ligne de moyenne pour la hauteur
    fig.add_trace(go.Scatter(
        x=[segment_data['DateTime'].iloc[0], segment_data['DateTime'].iloc[-1]],
        y=[hauteur_moyenne, hauteur_moyenne],
        mode='lines', line=dict(color='red', width=1.5, dash='dash'),
        name='Moyenne', showlegend=False
    ), row=2, col=1)
    
    # Longueur
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=segment_data['Length'],
        mode='markers', marker=dict(color='lightgray', size=6), name='Longueur (brut)',
        showlegend=False
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=smooth(segment_data['Length']),
        mode='lines', line=dict(color='blue', width=2), name='Longueur (lissé)',
        showlegend=False
    ), row=3, col=1)
    # Ligne de moyenne pour la longueur
    fig.add_trace(go.Scatter(
        x=[segment_data['DateTime'].iloc[0], segment_data['DateTime'].iloc[-1]],
        y=[longueur_moyenne, longueur_moyenne],
        mode='lines', line=dict(color='red', width=1.5, dash='dash'),
        name='Moyenne', showlegend=False
    ), row=3, col=1)
    
    # Ajout des noms complets sur l'axe Y
    fig.update_yaxes(title_text="Vitesse (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Hauteur (m)", row=2, col=1)
    fig.update_yaxes(title_text="Longueur (m)", row=3, col=1)
    
    # Titre global et informations
    fig.update_layout(
        title={
            'text': f"Analyse du segment de marche - {duree_minutes:.1f} min - {nb_pas} pas",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color="#2CC1AA")
        },
        height=600, 
        showlegend=False,
        margin=dict(t=80, b=40), 
        plot_bgcolor="white", 
        paper_bgcolor="white", 
        font=dict(family="Roboto, Arial, sans-serif", size=15)
    )
    
    # Ajouter des annotations pour les moyennes avec un meilleur formatage
    fig.add_annotation(
        x=segment_data['DateTime'].iloc[-1], 
        y=vitesse_moyenne,
        text=f"Moy: {vitesse_moyenne:.2f} m/s", 
        xanchor="right", 
        xshift=10,
        yshift=10,
        showarrow=False, 
        row=1, col=1, 
        font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1,
        borderpad=4
    )
    
    fig.add_annotation(
        x=segment_data['DateTime'].iloc[-1], 
        y=hauteur_moyenne,
        text=f"Moy: {hauteur_moyenne:.2f} m", 
        xanchor="right", 
        xshift=10,
        yshift=10,
        showarrow=False, 
        row=2, col=1, 
        font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1,
        borderpad=4
    )
    
    fig.add_annotation(
        x=segment_data['DateTime'].iloc[-1], 
        y=longueur_moyenne,
        text=f"Moy: {longueur_moyenne:.2f} m", 
        xanchor="right", 
        xshift=10,
        yshift=10,
        showarrow=False, 
        row=3, col=1, 
        font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1,
        borderpad=4
    )
    
    return fig

def make_daily_bar_figure(df):
    # Cas où aucune donnée n'est disponible
    if len(daily_stats) == 0:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(
                text="Aucune donnée disponible",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                font=dict(size=20)
            )],
            height=420,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Roboto, Arial, sans-serif", size=15)
        )
        return fig
        
    colors = {
        'steps': '#2CC1AA',    # Turquoise
        'distance': '#E7174A',  # Rouge
        'time': '#FF9A16'      # Orange
    }
    fig = go.Figure()

    # Convertir les dates au format datetime pour tri chronologique
    date_objects = []
    for date_str in dates:
        try:
            # Format DD/MM/YYYY
            parts = date_str.split('/')
            if len(parts) == 3:
                day, month, year = parts
                date_obj = datetime.datetime(int(year), int(month), int(day))
                date_objects.append((date_str, date_obj))
        except:
            # Si la conversion échoue, ajouter une date arbitraire
            date_objects.append((date_str, datetime.datetime(1900, 1, 1)))
    
    # Trier par date
    date_objects.sort(key=lambda x: x[1])
    sorted_dates = [date_tuple[0] for date_tuple in date_objects]

    # Préparer les données pour le graphique
    steps_values = [daily_stats[d]['steps'] for d in sorted_dates]
    distance_values = [daily_stats[d]['distance'] for d in sorted_dates]
    time_values = [daily_stats[d]['time'] for d in sorted_dates]

    # Barres principales (pas et distance) sur axe primaire
    fig.add_trace(go.Bar(
        x=sorted_dates, y=steps_values,
        name='Nombre de pas', marker_color=colors['steps'],
        text=[int(v) for v in steps_values], textposition='auto', yaxis='y'
    ))
    fig.add_trace(go.Bar(
        x=sorted_dates, y=distance_values,
        name='Distance (m)', marker_color=colors['distance'],
        text=[int(v) for v in distance_values], textposition='auto', yaxis='y'
    ))
    
    # Temps de marche sur axe secondaire, en ligne
    fig.add_trace(go.Scatter(
        x=sorted_dates, y=time_values,
        name='Temps de marche (min)', mode='lines+markers+text',
        marker=dict(color=colors['time'], size=10),
        line=dict(color=colors['time'], width=3),
        text=[int(v) for v in time_values],
        textposition='top center',
        yaxis='y2'
    ))

    fig.update_layout(
        barmode='group',
        xaxis=dict(
            showgrid=False,
            showticklabels=True,   # Affiche les labels de l'axe X
            showline=True,         # Affiche la ligne de l'axe X
            zeroline=False,
            title=None
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            title=None
        ),
        yaxis2=dict(
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            title=None,
            overlaying='y',
            side='right'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=40, b=60, l=40, r=40),
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Roboto, Arial, sans-serif", size=15)
    )
    return fig

def make_analysis_content(df):
    global six_min_test_data
    
    # Obtenir les valeurs du test de 6 minutes si disponibles
    six_min_speed = six_min_test_data['Speed'].mean() if six_min_test_data is not None else None
    six_min_length = six_min_test_data['Length'].mean() if six_min_test_data is not None else None
    six_min_height = six_min_test_data['Height'].mean() if six_min_test_data is not None else None
    six_min_steps = len(six_min_test_data) * 2 if six_min_test_data is not None else None
    six_min_distance = len(six_min_test_data) * six_min_length if six_min_test_data is not None else None
    
    # Utiliser les valeurs du test de 6 minutes comme valeurs secondaires (total)
    return html.Div([
        html.Div([
            indicator_bar(df['Speed'].mean(), 0, 2, "Vitesse de pas (m/s)", six_min_speed),
            indicator_bar(df['Length'].mean(), 0, 2, "Longueur de pas (m)", six_min_length),
            indicator_bar(df['Height'].mean(), 0, 0.2, "Hauteur de pas (m)", six_min_height),
            # Ajuster l'échelle pour le nombre de pas (pour l'affichage quotidien)
            # Cela permet au test de 6 minutes de ne pas toujours être dans la zone "Mauvais"
            indicator_bar(len(df), 0, 15000, "Nombre de pas", six_min_steps),
            # Ajuster l'échelle pour la distance parcourue (pour l'affichage quotidien)
            indicator_bar(int(len(df) * df['Length'].mean()), 0, 15000, "Distance parcourue (m)", six_min_distance),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "1.2em",
            "justifyContent": "center",
            "maxWidth": "900px",
            "margin": "0 auto"
        }),
    ])

from dash import dash_table
from plotly.subplots import make_subplots

def login_form():
    input_style = {
        "marginBottom": "1em",
        "width": "100%",
        "padding": "0.9em",
        "borderRadius": "7px",
        "border": "1.5px solid #e0e0e0",
        "fontSize": "1.1em"
    }
    return html.Div([
        html.Div([
            html.Div("🦵", style={"fontSize": "2.8em", "textAlign": "center", "marginBottom": "0.5em"}),
            html.H2("Connexion", style={"color": "#2CC1AA", "marginBottom": "1.2em", "textAlign": "center", "fontWeight": "bold"}),
            dcc.Input(id="login-username", type="text", placeholder="Identifiant", style=input_style),
            dcc.Input(id="login-password", type="password", placeholder="Mot de passe", style=input_style),
            html.Button("Se connecter", id="login-button", n_clicks=0, style={
                "width": "100%", "background": "#2CC1AA", "color": "white", "border": "none",
                "padding": "0.9em", "borderRadius": "7px", "fontWeight": "bold", "fontSize": "1.1em", "boxShadow": "0 2px 8px rgba(44,193,170,0.07)"
            }),
            html.Div(id="login-error", style={"color": "red", "marginTop": "1em", "textAlign": "center"})
        ], style={
            "maxWidth": "370px", "margin": "8em auto", "background": "white", "padding": "2.5em 2.5em",
            "borderRadius": "18px", "boxShadow": "0 4px 32px rgba(44,193,170,0.10)"
        })
    ], style={"background": "#f7f9fa", "minHeight": "100vh"})

def main_app_layout():
    return html.Div([
        html.Div([
            html.Span("🦵", style={"fontSize": "2em", "marginRight": "0.5em"}),
            html.Span("Compte Rendu | Analyse de la Marche", style={"fontWeight": "bold", "fontSize": "1.6em"}),
        ], className="header-bar"),
        html.Div(
            html.Button([
                "Déconnexion"
            ], id="logout-button", n_clicks=0, className="logout-btn"),
            style={"display": "flex", "justifyContent": "center", "margin": "1.2em 0 2.2em 0"}
        ),

        # Sélecteur de patient existant
        html.Div([
            html.Label("Sélectionner une fiche patient existante :", style={"fontWeight": "bold", "color": "#2CC1AA"}),
            dcc.Dropdown(
                id="patient-select",
                options=[{"label": k.replace("_", " "), "value": k} for k in list_patients()],
                placeholder="Choisir un patient...",
                style={"marginBottom": "1.5em"}
            ),
        ], className="card", id="select-patient-card"),

        # Formulaire de création (masqué si patient sélectionné)
        html.Div([
            html.Div("Créer une fiche patient", className="form-title"),
            html.Div([
                html.Div([html.Label("Prénom"), dcc.Input(id="prenom", type="text", placeholder="Prénom")], className="form-group"),
                html.Div([html.Label("Nom"), dcc.Input(id="nom", type="text", placeholder="Nom")], className="form-group"),
                html.Div([html.Label("Âge"), dcc.Input(id="age", type="number", placeholder="Âge")], className="form-group"),
                html.Div([html.Label("Taille (cm)"), dcc.Input(id="taille", type="number", placeholder="Taille")], className="form-group"),
                html.Div([html.Label("Poids (kg)"), dcc.Input(id="poids", type="number", placeholder="Poids")], className="form-group"),
                html.Div([html.Label("Pathologie"), dcc.Input(id="patho", type="text", placeholder="Pathologie")], className="form-group"),
            ], className="form-row"),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Uploader le CSV'),
                multiple=False,
                className="dash-uploader"
            ),
            html.Button("Valider", id="submit-patient", n_clicks=0),
            html.Div(id="patient-feedback", style={"marginTop": "1em", "color": "#2CC1AA", "fontWeight": "bold"})
        ], className="card", id="create-patient-card"),

        html.Div([
            html.Div(id="patient-summary", style={
                "flex": "0 0 320px",
                "maxWidth": "320px",
                "marginRight": "2.5em"
            }),
            html.Div(id="analysis-section", style={
                "flex": "1 1 0",
                "minWidth": "320px",
                "width": "100%"
            }),
        ], style={
            "display": "flex",
            "alignItems": "flex-start",
            "justifyContent": "center",
            "gap": "2.5em",
            "margin": "2em 0 2em 0",
            "width": "100%"
        }),

        # Section Test de marche de 6 minutes - maintenant avant la synthèse quotidienne
        html.Div([
            html.Hr(style={"margin": "2em 0"}),
            html.H4([
                html.Span("Test de marche de 6 minutes", style={
                    "fontWeight": "bold",
                    "fontSize": "1.25em",
                    "letterSpacing": "0.01em",
                    "color": "#2CC1AA",
                    "background": "rgba(44,193,170,0.07)",
                    "padding": "0.4em 1.2em",
                    "borderRadius": "8px",
                    "boxShadow": "0 1px 6px rgba(44,193,170,0.07)"
                })
            ], style={"margin": "2em 0 1em 0", "textAlign": "left"}),
            
            # Zone d'upload - clairement séparée du graphique
            html.Div([
                dcc.Upload(
                    id='upload-six-min-test',
                    children=html.Div([
                        'Glisser-déposer ou ',
                        html.A('sélectionner un fichier CSV', style={"color": "#2CC1AA", "textDecoration": "underline"})
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0',
                        'position': 'relative',  # Position relative pour éviter le chevauchement
                        'zIndex': '10'  # Z-index plus élevé pour s'assurer qu'il reste au-dessus
                    },
                    multiple=False
                ),
                html.Div(id='six-min-test-upload-output', style={'marginBottom': '20px'})
            ], style={"margin": "1em 0", "position": "relative"}),
            
            # Graphique - maintenant avec une marge claire par rapport à la zone d'upload
            html.Div(id='six-min-test-graph', style={"marginTop": "40px"})
        ], id="six-min-test-section", style={"display": "none"}),
        
        # Graphiques de synthèse et activité quotidienne
        html.Div(id="graphs-section", style={"width": "100%", "margin": "0 auto"}),
    ], style={"maxWidth": "1200px", "margin": "auto", "padding": "1em"})

app.layout = html.Div([
    dcc.Store(id="login-state", storage_type="session"),
    html.Div(id="login-container", children=login_form()),
    html.Div(id="main-app", style={"display": "none"}),
    html.Button("Déconnexion", id="logout-button", n_clicks=0, style={"display": "none"})
])

@app.callback(
    Output("login-container", "children"),
    Output("main-app", "children"),
    Output("main-app", "style"),
    Output("login-state", "data"),
    Input("login-button", "n_clicks"),
    Input("logout-button", "n_clicks"),
    State("login-username", "value"),
    State("login-password", "value"),
    State("login-state", "data"),
    prevent_initial_call=True
)
def handle_login_and_display(n_clicks_login, n_clicks_logout, username, password, login_state):
    ctx = dash.callback_context

    # Déconnexion
    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith("logout-button"):
        return login_form(), "", {"display": "none"}, None

    # Connexion
    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith("login-button"):
        if username in VALID_USERS and password == VALID_USERS[username]:
            return "", main_app_layout(), {"display": "block"}, "ok"
        else:
            return login_form() + [html.Div("Identifiant ou mot de passe incorrect.", id="login-error", style={"color": "red", "marginTop": "1em"})], "", {"display": "none"}, None

    # Si déjà connecté (après refresh ou navigation)
    if login_state == "ok":
        return "", main_app_layout(), {"display": "block"}, "ok"

    # Sinon (premier affichage, pas connecté)
    return login_form(), "", {"display": "none"}, None

@app.callback(
    Output("segment-detail", "children"),
    Input("density-plot", "clickData")
)
def show_segment_detail(clickData):
    if not clickData or "points" not in clickData or not clickData["points"] or "customdata" not in clickData["points"][0] or clickData["points"][0]["customdata"] is None:
        # Message par défaut - juste un espace vide
        return html.Div(style={"height": "20px"})
    
    try:
        # Récupérer le segment cliqué
        customdata = clickData["points"][0]["customdata"]
        file_name, start_time, end_time = customdata.split("|")
        start_time = float(start_time)
        end_time = float(end_time)
        # Retrouver le segment
        for seg in segments:
            if seg["file_name"] == file_name and abs(seg["start_time"] - start_time) < 1e-4 and abs(seg["end_time"] - end_time) < 1e-4:
                segment_data = seg["segment_data"]
                break
        else:
            # Segment non trouvé
            return html.Div("Segment non trouvé ou trop court pour être analysé.", style={"margin": "2em", "textAlign": "center"})
        
        # Graphique détaillé avec statistiques intégrées
        fig = make_detail_figure(segment_data)
        
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Erreur lors de l'affichage des détails: {str(e)}", style={"margin": "2em", "textAlign": "center"})

@app.callback(
    Output("create-patient-card", "style"),
    Output("select-patient-card", "style"),
    Output("patient-summary", "children"),
    Output("analysis-section", "children"),
    Output("patient-feedback", "children"),
    Output("patient-select", "options"),
    Output("graphs-section", "children"),
    Output('six-min-test-upload-output', 'children', allow_duplicate=True),
    Output('six-min-test-graph', 'children', allow_duplicate=True),
    Input("submit-patient", "n_clicks"),
    Input("patient-select", "value"),
    State("prenom", "value"),
    State("nom", "value"),
    State("age", "value"),
    State("taille", "value"),
    State("poids", "value"),
    State("patho", "value"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def handle_patient(n_clicks, selected_patient, prenom, nom, age, taille, poids, patho, contents, filename):
    global segments, daily_stats, dates, six_min_test_data
    ctx = dash.callback_context
    # Si sélection d'un patient existant
    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith("patient-select"):
        if selected_patient:
            infos, df = load_patient(selected_patient)
            
            # Charger automatiquement le test de 6 minutes s'il existe
            six_min_test_data = load_six_min_test(selected_patient)
            six_min_test_graph = html.Div()
            if six_min_test_data is not None:
                six_min_test_graph = dcc.Graph(figure=make_six_min_test_figure(six_min_test_data))
            
            # Recalculer les segments pour ce patient
            segments = []
            for file_name in df['FileName'].unique():
                file_data = df[df['FileName'] == file_name].sort_values('DateTime')
                time_diff = file_data['HourOfDay'].diff()
                new_segment = (time_diff > 0.0083).astype(int).cumsum()
                for segment in new_segment.unique():
                    segment_data = file_data[new_segment == segment]
                    if len(segment_data) > 0:
                        start_time = segment_data['HourOfDay'].iloc[0]
                        end_time = segment_data['HourOfDay'].iloc[-1]
                        duration = (end_time - start_time) * 60
                        segments.append({
                            "file_name": file_name,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": duration,
                            "segment_data": segment_data
                        })
            
            # Recalculer les statistiques quotidiennes
            daily_stats = {}
            for file_name in df['FileName'].unique():
                file_data = df[df['FileName'] == file_name]
                # Nombre de pas (nombre total de mesures multiplié par 2)
                nb_steps = len(file_data) * 2
                # Distance totale (somme des longueurs)
                total_distance = file_data['Length'].sum()
                # Temps de marche (en minutes)
                walking_time = file_data['WalkingMinutes'].iloc[0] if 'WalkingMinutes' in file_data.columns and len(file_data) > 0 else 0
                # Formater la date pour l'affichage
                try:
                    date_str = ''.join(filter(str.isdigit, file_name))[:8]
                    formatted_date = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]}"
                except:
                    formatted_date = file_name
                daily_stats[formatted_date] = {
                    'steps': nb_steps,
                    'distance': total_distance,
                    'time': walking_time
                }
            dates = sorted(daily_stats.keys())
            
            summary = make_patient_summary(infos)
            analysis = make_analysis_content(df)
            graphs = html.Div([
                html.Hr(style={"margin": "2em 0"}),
                html.H4([
                    html.Span("Synthèse quotidienne de la marche active", style={
                        "fontWeight": "bold",
                        "fontSize": "1.25em",
                        "letterSpacing": "0.01em",
                        "color": "#2CC1AA",
                        "background": "rgba(44,193,170,0.07)",
                        "padding": "0.4em 1.2em",
                        "borderRadius": "8px",
                        "boxShadow": "0 1px 6px rgba(44,193,170,0.07)"
                    })
                ], style={"margin": "2em 0 1em 0", "textAlign": "left"}),
                dcc.Graph(figure=make_daily_bar_figure(df), style={"height": "500px", "background": "white", "borderRadius": "18px", "maxWidth": "1100px", "margin": "auto"}),
                
                # Espace entre les graphiques
                html.Div(style={"height": "3em"}),
                
                html.H4([
                    html.Span("Périodes exactes d'activité de marche", style={
                        "fontWeight": "bold",
                        "fontSize": "1.25em",
                        "letterSpacing": "0.01em",
                        "color": "#2CC1AA",
                        "background": "rgba(44,193,170,0.07)",
                        "padding": "0.4em 1.2em",
                        "borderRadius": "8px",
                        "boxShadow": "0 1px 6px rgba(44,193,170,0.07)"
                    })
                ], style={"margin": "2em 0 1em 0", "textAlign": "left"}),
                html.Div([
                    html.Div("Cliquez sur un segment d'activité (durée min. 2 min) pour voir les détails", style={
                        "fontStyle": "italic",
                        "color": "#666",
                        "margin": "0.5em 0 1.5em 0",
                        "textAlign": "center"
                    }),
                    dcc.Graph(id="density-plot", figure=make_density_figure(df), style={"maxWidth": "1100px", "margin": "auto"})
                ]),
                
                # Espace avant le détail du segment
                html.Div(style={"height": "2em"}),
                
                html.Div(id="segment-detail")
            ], style={"width": "100%", "margin": "0 auto"})
            return (
                {"display": "none"},  # create-patient-card
                {"display": "block"}, # select-patient-card
                summary,              # patient-summary
                analysis,             # analysis-section (indicateurs de perf)
                "",                   # patient-feedback
                [{"label": k.replace("_", " "), "value": k} for k in list_patients()],  # patient-select options
                graphs,               # graphs-section (les deux autres graphes)
                "",                   # six-min-test-upload-output
                six_min_test_graph    # six-min-test-graph
            )
        else:
            return {"display": "block"}, {"display": "block"}, "", "", "", [{"label": k.replace("_", " "), "value": k} for k in list_patients()], html.Div(), "", html.Div()
    # Si création d'un nouveau patient
    if not (prenom and nom and age and taille and poids and patho and contents):
        return {"display": "block"}, {"display": "block"}, "", "", "Veuillez remplir tous les champs et uploader un CSV.", [{"label": k.replace("_", " "), "value": k} for k in list_patients()], html.Div(), "", html.Div()
    try:
        taille_m = float(taille) / 100
        poids_kg = float(poids)
        imc = poids_kg / (taille_m ** 2)
    except Exception:
        return {"display": "block"}, {"display": "block"}, "", "", "Erreur dans la saisie de la taille ou du poids.", [{"label": k.replace("_", " "), "value": k} for k in list_patients()], html.Div(), "", html.Div()
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df_local = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df_local['DateTime'] = pd.to_datetime(df_local['DateTime'])
    except Exception as e:
        return {"display": "block"}, {"display": "block"}, "", "", f"Erreur lors de la lecture du CSV : {e}", [{"label": k.replace("_", " "), "value": k} for k in list_patients()], html.Div(), "", html.Div()
    infos = {
        "prenom": prenom, "nom": nom, "age": age, "taille": taille, "poids": poids, "imc": round(imc, 1), "patho": patho
    }
    save_patient(prenom, nom, infos, df_local)
    
    # Réinitialiser les données du test de 6 minutes
    six_min_test_data = None
    
    # Recalculer les segments pour ce patient
    segments = []
    for file_name in df_local['FileName'].unique():
        file_data = df_local[df_local['FileName'] == file_name].sort_values('DateTime')
        time_diff = file_data['HourOfDay'].diff()
        new_segment = (time_diff > 0.0083).astype(int).cumsum()
        for segment in new_segment.unique():
            segment_data = file_data[new_segment == segment]
            if len(segment_data) > 0:
                start_time = segment_data['HourOfDay'].iloc[0]
                end_time = segment_data['HourOfDay'].iloc[-1]
                duration = (end_time - start_time) * 60
                segments.append({
                    "file_name": file_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "segment_data": segment_data
                })
    
    # Recalculer les statistiques quotidiennes
    daily_stats = {}
    for file_name in df_local['FileName'].unique():
        file_data = df_local[df_local['FileName'] == file_name]
        # Nombre de pas (nombre total de mesures multiplié par 2)
        nb_steps = len(file_data) * 2
        # Distance totale (somme des longueurs)
        total_distance = file_data['Length'].sum()
        # Temps de marche (en minutes)
        walking_time = file_data['WalkingMinutes'].iloc[0] if 'WalkingMinutes' in file_data.columns and len(file_data) > 0 else 0
        # Formater la date pour l'affichage
        try:
            date_str = ''.join(filter(str.isdigit, file_name))[:8]
            formatted_date = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]}"
        except:
            formatted_date = file_name
        daily_stats[formatted_date] = {
            'steps': nb_steps,
            'distance': total_distance,
            'time': walking_time
        }
    dates = sorted(daily_stats.keys())
    
    summary = make_patient_summary(infos)
    analysis = make_analysis_content(df_local)
    graphs = html.Div([
        html.Hr(style={"margin": "2em 0"}),
        html.H4([
            html.Span("Synthèse quotidienne de la marche active", style={
                "fontWeight": "bold",
                "fontSize": "1.25em",
                "letterSpacing": "0.01em",
                "color": "#2CC1AA",
                "background": "rgba(44,193,170,0.07)",
                "padding": "0.4em 1.2em",
                "borderRadius": "8px",
                "boxShadow": "0 1px 6px rgba(44,193,170,0.07)"
            })
        ], style={"margin": "2em 0 1em 0", "textAlign": "left"}),
        dcc.Graph(figure=make_daily_bar_figure(df_local), style={"height": "500px", "background": "white", "borderRadius": "18px", "maxWidth": "1100px", "margin": "auto"}),
        
        # Espace entre les graphiques
        html.Div(style={"height": "3em"}),
        
        html.H4([
            html.Span("Périodes exactes d'activité de marche", style={
                "fontWeight": "bold",
                "fontSize": "1.25em",
                "letterSpacing": "0.01em",
                "color": "#2CC1AA",
                "background": "rgba(44,193,170,0.07)",
                "padding": "0.4em 1.2em",
                "borderRadius": "8px",
                "boxShadow": "0 1px 6px rgba(44,193,170,0.07)"
            })
        ], style={"margin": "2em 0 1em 0", "textAlign": "left"}),
        html.Div([
            html.Div("Cliquez sur un segment d'activité (durée min. 2 min) pour voir les détails", style={
                "fontStyle": "italic",
                "color": "#666",
                "margin": "0.5em 0 1.5em 0",
                "textAlign": "center"
            }),
            dcc.Graph(id="density-plot", figure=make_density_figure(df_local), style={"maxWidth": "1100px", "margin": "auto"})
        ]),
        
        # Espace avant le détail du segment
        html.Div(style={"height": "2em"}),
        
        html.Div(id="segment-detail")
    ], style={"width": "100%", "margin": "0 auto"})
    return (
        {"display": "none"},  # create-patient-card
        {"display": "block"}, # select-patient-card
        summary,              # patient-summary
        analysis,             # analysis-section (indicateurs de perf)
        "",                   # patient-feedback
        [{"label": k.replace("_", " "), "value": k} for k in list_patients()],  # patient-select options
        graphs,               # graphs-section (les deux autres graphes)
        "",                   # six-min-test-upload-output
        html.Div()            # six-min-test-graph
    )

def make_patient_summary(infos):
    return html.Div([
        html.Div([
            html.Span("🧑‍⚕️", style={"fontSize": "1.5em", "marginRight": "0.5em"}),
            html.Span("Fiche patient", style={"color": "#2CC1AA", "fontWeight": "bold", "fontSize": "1.2em"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "0.5em"}),
        html.Hr(style={"border": "none", "borderTop": "1.5px solid #e0e0e0", "margin": "0 0 1.2em 0"}),
        html.Div([
            row("Nom", f"{infos['prenom']} {infos['nom']}"),
            row("Âge", f"{infos['age']} ans"),
            row("Taille", f"{infos['taille']} cm"),
            row("Poids", f"{infos['poids']} kg"),
            row("IMC", f"{infos['imc']}"),
            row("Pathologie", f"{infos['patho']}"),
        ], style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "1.1em",
            "fontSize": "1.13em",
            "lineHeight": "1.9"
        }),
    ], className="card", style={
        "maxWidth": "500px",
        "margin": "2em auto 2em auto",
        "boxShadow": "0 2px 16px rgba(44,193,170,0.10)",
        "padding": "2em 2.2em"
    })

def row(label, value):
    return html.Div([
        html.Span(f"{label} :", style={"color": "#2CC1AA", "fontWeight": "bold", "minWidth": "110px", "display": "inline-block"}),
        html.Span(value, style={"marginLeft": "1em"})
    ], style={"display": "flex", "justifyContent": "space-between"})

def make_six_min_test_figure(df):
    # Vérifier si les données sont vides
    if df is None or len(df) == 0:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(
                text="Aucune donnée disponible",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                font=dict(size=20)
            )],
            height=600,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Roboto, Arial, sans-serif", size=15)
        )
        return fig
        
    def smooth(y, window=15):
        return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean()
    
    # Calculer les moyennes et les statistiques
    vitesse_moyenne = df['Speed'].mean()
    hauteur_moyenne = df['Height'].mean()
    longueur_moyenne = df['Length'].mean()
    
    # Créer les sous-graphiques sans titres
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    
    # Vitesse
    fig.add_trace(go.Scatter(
        x=df['DateTime'], y=df['Speed'],
        mode='markers', marker=dict(color='lightgray', size=6), name='Vitesse (brut)',
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['DateTime'], y=smooth(df['Speed']),
        mode='lines', line=dict(color='blue', width=2), name='Vitesse (lissé)',
        showlegend=False
    ), row=1, col=1)
    # Ligne de moyenne pour la vitesse
    fig.add_trace(go.Scatter(
        x=[df['DateTime'].iloc[0], df['DateTime'].iloc[-1]],
        y=[vitesse_moyenne, vitesse_moyenne],
        mode='lines', line=dict(color='red', width=1.5, dash='dash'),
        name='Moyenne', showlegend=False
    ), row=1, col=1)
    
    # Hauteur
    fig.add_trace(go.Scatter(
        x=df['DateTime'], y=df['Height'],
        mode='markers', marker=dict(color='lightgray', size=6), name='Hauteur (brut)',
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df['DateTime'], y=smooth(df['Height']),
        mode='lines', line=dict(color='blue', width=2), name='Hauteur (lissé)',
        showlegend=False
    ), row=2, col=1)
    # Ligne de moyenne pour la hauteur
    fig.add_trace(go.Scatter(
        x=[df['DateTime'].iloc[0], df['DateTime'].iloc[-1]],
        y=[hauteur_moyenne, hauteur_moyenne],
        mode='lines', line=dict(color='red', width=1.5, dash='dash'),
        name='Moyenne', showlegend=False
    ), row=2, col=1)
    
    # Longueur
    fig.add_trace(go.Scatter(
        x=df['DateTime'], y=df['Length'],
        mode='markers', marker=dict(color='lightgray', size=6), name='Longueur (brut)',
        showlegend=False
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df['DateTime'], y=smooth(df['Length']),
        mode='lines', line=dict(color='blue', width=2), name='Longueur (lissé)',
        showlegend=False
    ), row=3, col=1)
    # Ligne de moyenne pour la longueur
    fig.add_trace(go.Scatter(
        x=[df['DateTime'].iloc[0], df['DateTime'].iloc[-1]],
        y=[longueur_moyenne, longueur_moyenne],
        mode='lines', line=dict(color='red', width=1.5, dash='dash'),
        name='Moyenne', showlegend=False
    ), row=3, col=1)
    
    # Ajout des noms complets sur l'axe Y
    fig.update_yaxes(title_text="Vitesse (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Hauteur (m)", row=2, col=1)
    fig.update_yaxes(title_text="Longueur (m)", row=3, col=1)
    
    # Titre global et informations
    fig.update_layout(
        title={
            'text': f"Test de marche de 6 minutes",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color="#2CC1AA")
        },
        height=600, 
        showlegend=False,
        margin=dict(t=80, b=40), 
        plot_bgcolor="white", 
        paper_bgcolor="white", 
        font=dict(family="Roboto, Arial, sans-serif", size=15)
    )
    
    # Ajouter des annotations pour les moyennes avec un meilleur formatage
    fig.add_annotation(
        x=df['DateTime'].iloc[-1], 
        y=vitesse_moyenne,
        text=f"Moy: {vitesse_moyenne:.2f} m/s", 
        xanchor="right", 
        xshift=10,
        yshift=10,
        showarrow=False, 
        row=1, col=1, 
        font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1,
        borderpad=4
    )
    
    fig.add_annotation(
        x=df['DateTime'].iloc[-1], 
        y=hauteur_moyenne,
        text=f"Moy: {hauteur_moyenne:.2f} m", 
        xanchor="right", 
        xshift=10,
        yshift=10,
        showarrow=False, 
        row=2, col=1, 
        font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1,
        borderpad=4
    )
    
    fig.add_annotation(
        x=df['DateTime'].iloc[-1], 
        y=longueur_moyenne,
        text=f"Moy: {longueur_moyenne:.2f} m", 
        xanchor="right", 
        xshift=10,
        yshift=10,
        showarrow=False, 
        row=3, col=1, 
        font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1,
        borderpad=4
    )
    
    return fig

@app.callback(
    Output("six-min-test-section", "style"),
    Output("upload-six-min-test", "children"),
    Input("patient-select", "value")
)
def toggle_six_min_test_section(selected_patient):
    upload_content = html.Div([
        'Glisser-déposer ou ',
        html.A('sélectionner un fichier CSV', style={"color": "#2CC1AA", "textDecoration": "underline"})
    ])
    
    if selected_patient:
        return {"display": "block"}, upload_content
    return {"display": "none"}, upload_content

@app.callback(
    Output('six-min-test-upload-output', 'children'),
    Output('six-min-test-graph', 'children'),
    Output('analysis-section', 'children', allow_duplicate=True),
    Input('upload-six-min-test', 'contents'),
    State('upload-six-min-test', 'filename'),
    State('patient-select', 'value'),  # Besoin du patient sélectionné pour charger les données
    prevent_initial_call=True
)
def update_six_min_test_output(contents, filename, selected_patient):
    global six_min_test_data
    
    if contents is None:
        return "", html.Div(), dash.no_update
        
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Charger le CSV
        six_min_test_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        six_min_test_data['DateTime'] = pd.to_datetime(six_min_test_data['DateTime'])
        
        # Sauvegarder le test pour le patient sélectionné
        if selected_patient:
            save_six_min_test(selected_patient, six_min_test_data)
            message = "Test de 6 minutes sauvegardé avec succès"
        else:
            message = "Attention: Patient non sélectionné, le test n'a pas été sauvegardé"
        
        # Créer le graphique
        graph = dcc.Graph(figure=make_six_min_test_figure(six_min_test_data))
        
        # Mettre à jour les indicateurs avec les données du patient sélectionné
        if selected_patient:
            infos, df = load_patient(selected_patient)
            updated_analysis = make_analysis_content(df)
            return message, graph, updated_analysis
        
        return message, graph, dash.no_update
    except Exception as e:
        return html.Div([
            "Une erreur est survenue lors du traitement du fichier.",
            html.Br(),
            str(e)
        ], style={"color": "red"}), html.Div(), dash.no_update

if __name__ == "__main__":
    app.run(debug=True)
