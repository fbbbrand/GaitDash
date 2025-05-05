import os
import json
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import base64
import io
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

# Utiliser le chemin relatif au fichier app.py
PATIENTS_DIR = os.path.join(os.path.dirname(__file__), "patients")
os.makedirs(PATIENTS_DIR, exist_ok=True)

# Variable globale pour stocker les données des patients
PATIENTS_DATA = {}

def load_all_patients():
    """Charge les données de tous les patients au démarrage"""
    global PATIENTS_DATA
    PATIENTS_DATA.clear()
    print(f"Chargement des patients depuis : {PATIENTS_DIR}")
    print(f"Fichiers trouvés : {os.listdir(PATIENTS_DIR)}")
    
    for filename in os.listdir(PATIENTS_DIR):
        if filename.endswith('.json'):
            patient_id = filename[:-5]  # Enlever l'extension .json
            try:
                print(f"Chargement du patient : {patient_id}")
                # Charger les infos du patient
                with open(os.path.join(PATIENTS_DIR, filename), 'r', encoding='utf-8') as f:
                    patient_info = json.load(f)
                # Charger les données CSV si elles existent
                csv_path = os.path.join(PATIENTS_DIR, f"{patient_id}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df['DateTime'] = pd.to_datetime(df['DateTime'])
                    PATIENTS_DATA[patient_id] = {
                        'info': patient_info,
                        'data': df
                    }
                    print(f"Patient {patient_id} chargé avec succès")
                else:
                    print(f"Fichier CSV manquant pour {patient_id}")
            except Exception as e:
                print(f"Erreur lors du chargement du patient {patient_id}: {e}")

# Charger les patients au démarrage
load_all_patients()

# Identifiants autorisés (à adapter)
VALID_USERS = {"admin": "motdepasse123"}

def list_patients():
    """Retourne la liste des patients disponibles"""
    print(f"Liste des patients disponibles : {list(PATIENTS_DATA.keys())}")
    return [{"label": k.replace("_", " "), "value": k} for k in PATIENTS_DATA.keys()]

def save_patient(prenom, nom, infos, df):
    global PATIENTS_DATA
    key = f"{prenom}_{nom}".replace(" ", "_")
    # Sauvegarde infos
    with open(os.path.join(PATIENTS_DIR, f"{key}.json"), "w", encoding="utf-8") as f:
        json.dump(infos, f)
    # Sauvegarde CSV
    df.to_csv(os.path.join(PATIENTS_DIR, f"{key}.csv"), index=False)
    # Mettre à jour les données en mémoire
    PATIENTS_DATA[key] = {
        'info': infos,
        'data': df
    }

def load_patient(key):
    global PATIENTS_DATA
    if key not in PATIENTS_DATA:
        # Si le patient n'est pas en mémoire, essayer de le charger
        try:
            with open(os.path.join(PATIENTS_DIR, f"{key}.json"), encoding="utf-8") as f:
                infos = json.load(f)
            df = pd.read_csv(os.path.join(PATIENTS_DIR, f"{key}.csv"))
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            PATIENTS_DATA[key] = {
                'info': infos,
                'data': df
            }
        except Exception as e:
            print(f"Erreur lors du chargement du patient {key}: {e}")
            return None, None
    return PATIENTS_DATA[key]['info'], PATIENTS_DATA[key]['data']

# Charger les données
csv_path = os.path.join(os.path.dirname(__file__), "walking_data_analysis_caen.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
else:
    df = None  # Ou gérez le cas où il n'y a pas de CSV par défaut

# --- Infos patient (exemple, à adapter selon ton CSV) ---
patient_info = {}

# --- Indicateurs de performance ---
perf_data = df.agg({'Speed': 'mean', 'Length': 'mean', 'Height': 'mean'})
nb_jours = df['FileName'].nunique()
nb_pas_total = len(df)
nb_pas_moyen = nb_pas_total / nb_jours
distance_totale = len(df) * perf_data['Length']
distance_moyenne = distance_totale / nb_jours

metrics = [
    ("Vitesse de pas (m/s)", perf_data['Speed']),
    ("Longueur de pas (m)", perf_data['Length']),
    ("Hauteur de pas (m)", perf_data['Height']),
    ("Nombre de pas", nb_pas_moyen),
    ("Distance parcourue (m)", distance_moyenne)
]

# --- Préparation des segments pour le graphique de densité ---
segments = []
for file_name in df['FileName'].unique():
    file_data = df[df['FileName'] == file_name].sort_values('DateTime')
    y_pos = file_name
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

# --- Statistiques quotidiennes ---
daily_stats = {}
for file_name in df['FileName'].unique():
    file_data = df[df['FileName'] == file_name]
    # Nombre de pas (nombre total de mesures multiplié par 2)
    nb_steps = len(file_data) * 2
    # Distance totale (somme des longueurs)
    total_distance = file_data['Length'].sum()
    # Temps de marche (en minutes)
    walking_time = file_data['WalkingMinutes'].iloc[0]
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

# --- Dash app ---
app = dash.Dash(__name__)
server = app.server
app.title = "Gait Analysis Dash"
app.config.suppress_callback_exceptions = True
#app._favicon = "🦵"


def indicator_bar(value, vmin, vmax, label, total=None):
    # Définir les bornes des zones
    zone1 = vmin + (vmax - vmin) / 3
    zone2 = vmin + 2 * (vmax - vmin) / 3
    percent = 100 * (value - vmin) / (vmax - vmin)
    percent = min(max(percent, 0), 100)
    # Couleurs des zones
    colors = ["#E7174A", "#FF9A16", "#2CC1AA"]  # Rouge, Orange, Vert
    zone_labels = ["Mauvais", "Bon", "Excellent"]
    return html.Div([
        html.Div([
            html.Span(f"{label}", style={"fontWeight": "bold", "fontSize": "1.1em"}),
            html.Br(),
            html.Span(f"{int(total)}", style={"fontWeight": "bold", "fontSize": "2.2em", "color": "#2CC1AA"})
            if total is not None else
            html.Span(f"{value:.2f}" if isinstance(value, float) else int(value),
                      style={"fontWeight": "bold", "fontSize": "2.2em", "color": "#2CC1AA"}),
            html.Br(),
            # Barre de fond avec 3 zones colorées
            html.Div([
                html.Div(style={
                    "width": "33.33%",
                    "height": "8px",
                    "background": colors[0],
                    "display": "inline-block",
                    "borderTopLeftRadius": "4px",
                    "borderBottomLeftRadius": "4px"
                }),
                html.Div(style={
                    "width": "33.33%",
                    "height": "8px",
                    "background": colors[1],
                    "display": "inline-block"
                }),
                html.Div(style={
                    "width": "33.34%",
                    "height": "8px",
                    "background": colors[2],
                    "display": "inline-block",
                    "borderTopRightRadius": "4px",
                    "borderBottomRightRadius": "4px"
                }),
                # Curseur valeur (petite ligne noire)
                html.Div(style={
                    "position": "absolute",
                    "left": f"calc({percent}% - 1px)",
                    "top": "0px",
                    "width": "2px",
                    "height": "14px",
                    "background": "black",
                    "zIndex": 2
                }),
            ], style={
                "position": "relative",
                "width": "100%",
                "marginTop": "0.5em",
                "height": "14px",
                "margin": "0.5em 0.5em 1.5em 0.5em"
            }),
            # Labels des zones
            html.Div([
                html.Span(zone_labels[0], style={"color": colors[0], "fontSize": "0.9em", "width": "33.33%", "display": "inline-block", "textAlign": "left"}),
                html.Span(zone_labels[1], style={"color": colors[1], "fontSize": "0.9em", "width": "33.33%", "display": "inline-block", "textAlign": "center"}),
                html.Span(zone_labels[2], style={"color": colors[2], "fontSize": "0.9em", "width": "33.34%", "display": "inline-block", "textAlign": "right"}),
            ], style={"width": "100%", "marginTop": "0.2em"}),
        ], style={"padding": "1em"})
    ], style={
        "background": "white",
        "borderRadius": "12px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
        "margin": "0.5em"
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
    def smooth(y, window=15):
        return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["Vitesse", "Hauteur", "Longueur"]
    )
    # Vitesse
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=segment_data['Speed'],
        mode='markers', marker=dict(color='lightgray', size=6), name='Vitesse (brut)'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=smooth(segment_data['Speed']),
        mode='lines', line=dict(color='blue', width=2), name='Vitesse (lissé)'
    ), row=1, col=1)
    # Hauteur
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=segment_data['Height'],
        mode='markers', marker=dict(color='lightgray', size=6), name='Hauteur (brut)'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=smooth(segment_data['Height']),
        mode='lines', line=dict(color='blue', width=2), name='Hauteur (lissé)'
    ), row=2, col=1)
    # Longueur
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=segment_data['Length'],
        mode='markers', marker=dict(color='lightgray', size=6), name='Longueur (brut)'
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=segment_data['DateTime'], y=smooth(segment_data['Length']),
        mode='lines', line=dict(color='blue', width=2), name='Longueur (lissé)'
    ), row=3, col=1)
    # Ajout des unités sur l'axe Y
    fig.update_yaxes(title_text="Vitesse (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Hauteur (m)", row=2, col=1)
    fig.update_yaxes(title_text="Longueur (m)", row=3, col=1)
    fig.update_layout(height=600, showlegend=False, margin=dict(t=40, b=40), plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Roboto, Arial, sans-serif", size=15))
    return fig

def make_daily_bar_figure(df):
    colors = {
        'steps': '#2CC1AA',    # Turquoise
        'distance': '#E7174A',  # Rouge
        'time': '#FF9A16'      # Orange
    }
    fig = go.Figure()

    # Barres principales (pas et distance) sur axe primaire
    fig.add_trace(go.Bar(
        x=dates, y=[daily_stats[d]['steps'] for d in dates],
        name='Nombre de pas', marker_color=colors['steps'],
        text=[int(daily_stats[d]['steps']) for d in dates], textposition='auto', yaxis='y'
    ))
    fig.add_trace(go.Bar(
        x=dates, y=[daily_stats[d]['distance'] for d in dates],
        name='Distance (m)', marker_color=colors['distance'],
        text=[int(daily_stats[d]['distance']) for d in dates], textposition='auto', yaxis='y'
    ))
    # Temps de marche sur axe secondaire, en ligne
    fig.add_trace(go.Scatter(
        x=dates, y=[daily_stats[d]['time'] for d in dates],
        name='Temps de marche (min)', mode='lines+markers+text',
        marker=dict(color=colors['time'], size=10),
        line=dict(color=colors['time'], width=3),
        text=[int(daily_stats[d]['time']) for d in dates],
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
        margin=dict(t=60, b=60, l=40, r=40),
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Roboto, Arial, sans-serif", size=15)
    )
    return fig

def make_analysis_content(df):
    return html.Div([
        html.Div([
            indicator_bar(df['Speed'].mean(), 0, 2, "Vitesse de pas (m/s)"),
            indicator_bar(df['Length'].mean(), 0, 2, "Longueur de pas (m)"),
            indicator_bar(df['Height'].mean(), 0, 0.2, "Hauteur de pas (m)"),
            indicator_bar(len(df), 0, 15000, "Nombre de pas"),
            indicator_bar(int(len(df) * df['Length'].mean()), 0, 15000, "Distance parcourue (m)"),
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
                options=list_patients(),
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
        html.Div(id="graphs-section", style={"width": "100%", "margin": "0 auto"})
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
    if not clickData or "points" not in clickData:
        return html.Div("Cliquez sur un segment pour voir le détail.", style={"marginTop": "2em"})
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
        return html.Div("Segment non trouvé.")
    fig = make_detail_figure(segment_data)
    return dcc.Graph(figure=fig)

@app.callback(
    Output("create-patient-card", "style"),
    Output("select-patient-card", "style"),
    Output("patient-summary", "children"),
    Output("analysis-section", "children"),
    Output("patient-feedback", "children"),
    Output("patient-select", "options"),
    Output("graphs-section", "children"),
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
    ctx = dash.callback_context
    # Si sélection d'un patient existant
    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith("patient-select"):
        if selected_patient:
            print(f"Patient sélectionné : {selected_patient}")
            infos, df = load_patient(selected_patient)
            if infos is None or df is None:
                return {"display": "block"}, {"display": "block"}, "", "", "Erreur lors du chargement du patient", list_patients(), html.Div()
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
                dcc.Graph(id="density-plot", figure=make_density_figure(df), style={"maxWidth": "1100px", "margin": "auto"}),
                html.Div(id="segment-detail")
            ], style={"width": "100%", "margin": "0 auto"})
            return (
                {"display": "none"},  # create-patient-card
                {"display": "block"}, # select-patient-card
                summary,              # patient-summary
                analysis,             # analysis-section (indicateurs de perf)
                "",                   # patient-feedback
                list_patients(),      # patient-select options
                graphs                # graphs-section (les deux autres graphes)
            )
        else:
            return {"display": "block"}, {"display": "block"}, "", "", "", list_patients(), html.Div()
    # Si création d'un nouveau patient
    if not (prenom and nom and age and taille and poids and patho and contents):
        return {"display": "block"}, {"display": "block"}, "", "", "Veuillez remplir tous les champs et uploader un CSV.", list_patients(), html.Div()
    try:
        taille_m = float(taille) / 100
        poids_kg = float(poids)
        imc = poids_kg / (taille_m ** 2)
    except Exception:
        return {"display": "block"}, {"display": "block"}, "", "", "Erreur dans la saisie de la taille ou du poids.", list_patients(), html.Div()
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df_local = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df_local['DateTime'] = pd.to_datetime(df_local['DateTime'])
    except Exception as e:
        return {"display": "block"}, {"display": "block"}, "", "", f"Erreur lors de la lecture du CSV : {e}", list_patients(), html.Div()
    infos = {
        "prenom": prenom, "nom": nom, "age": age, "taille": taille, "poids": poids, "imc": round(imc, 1), "patho": patho
    }
    save_patient(prenom, nom, infos, df_local)
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
        dcc.Graph(id="density-plot", figure=make_density_figure(df_local), style={"maxWidth": "1100px", "margin": "auto"}),
        html.Div(id="segment-detail")
    ], style={"width": "100%", "margin": "0 auto"})
    return (
        {"display": "none"},  # create-patient-card
        {"display": "block"}, # select-patient-card
        summary,              # patient-summary
        analysis,             # analysis-section (indicateurs de perf)
        "",                   # patient-feedback
        list_patients(),      # patient-select options
        graphs                # graphs-section (les deux autres graphes)
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

if __name__ == "__main__":
    app.run(debug=True)