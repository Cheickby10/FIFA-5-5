import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import os
import re
from collections import defaultdict, Counter
from datetime import datetime

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="⚽ FIFA Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= FONCTIONS DE SAUVEGARDE =================
def save_matches_to_file(match_list):
    """Sauvegarde les matchs dans un fichier JSON"""
    try:
        with open('match_history.json', 'w') as f:
            json.dump(match_list, f)
        return True
    except:
        return False

def load_matches_from_file():
    """Charge les matchs depuis le fichier JSON"""
    try:
        if os.path.exists('match_history.json'):
            with open('match_history.json', 'r') as f:
                return json.load(f)
    except:
        pass
    return None

# ================= DONNÉES FIFA (25 ÉQUIPES) =================
FIFA_TEAMS = {
    # Top Clubs Europe
    "Real Madrid": {"attack": 87, "defense": 85, "midfield": 86, "overall": 86, "style": "counter_attack"},
    "Barcelona": {"attack": 86, "defense": 84, "midfield": 87, "overall": 86, "style": "possession"},
    "Manchester City": {"attack": 88, "defense": 85, "midfield": 88, "overall": 87, "style": "possession"},
    "Liverpool": {"attack": 86, "defense": 84, "midfield": 85, "overall": 85, "style": "high_press"},
    "Bayern Munich": {"attack": 87, "defense": 85, "midfield": 86, "overall": 86, "style": "possession"},
    "PSG": {"attack": 87, "defense": 83, "midfield": 84, "overall": 85, "style": "attacking"},
    
    # Premier League
    "Arsenal": {"attack": 85, "defense": 83, "midfield": 84, "overall": 84, "style": "attacking"},
    "Chelsea": {"attack": 82, "defense": 84, "midfield": 83, "overall": 83, "style": "counter_attack"},
    "Manchester United": {"attack": 83, "defense": 82, "midfield": 83, "overall": 83, "style": "counter_attack"},
    "Tottenham": {"attack": 84, "defense": 82, "midfield": 83, "overall": 83, "style": "attacking"},
    "Newcastle": {"attack": 83, "defense": 84, "midfield": 82, "overall": 83, "style": "balanced"},
    "Aston Villa": {"attack": 84, "defense": 81, "midfield": 83, "overall": 83, "style": "attacking"},
    
    # La Liga
    "Atletico Madrid": {"attack": 84, "defense": 86, "midfield": 83, "overall": 84, "style": "defensive"},
    "Sevilla": {"attack": 82, "defense": 83, "midfield": 82, "overall": 82, "style": "balanced"},
    "Valencia": {"attack": 81, "defense": 82, "midfield": 81, "overall": 81, "style": "counter_attack"},
    "Villarreal": {"attack": 83, "defense": 82, "midfield": 84, "overall": 83, "style": "possession"},
    
    # Bundesliga
    "Dortmund": {"attack": 85, "defense": 81, "midfield": 84, "overall": 83, "style": "high_press"},
    "Leipzig": {"attack": 84, "defense": 82, "midfield": 84, "overall": 83, "style": "pressing"},
    "Leverkusen": {"attack": 84, "defense": 83, "midfield": 84, "overall": 84, "style": "possession"},
    
    # Serie A
    "Juventus": {"attack": 83, "defense": 85, "midfield": 83, "overall": 84, "style": "defensive"},
    "Milan": {"attack": 84, "defense": 83, "midfield": 82, "overall": 83, "style": "balanced"},
    "Inter": {"attack": 85, "defense": 84, "midfield": 84, "overall": 84, "style": "counter_attack"},
    "Napoli": {"attack": 85, "defense": 82, "midfield": 83, "overall": 83, "style": "attacking"},
    "Roma": {"attack": 83, "defense": 83, "midfield": 82, "overall": 83, "style": "balanced"},
    
    # Autres
    "Porto": {"attack": 82, "defense": 83, "midfield": 81, "overall": 82, "style": "defensive"},
    "Benfica": {"attack": 83, "defense": 81, "midfield": 83, "overall": 82, "style": "attacking"},
    "Ajax": {"attack": 82, "defense": 79, "midfield": 84, "overall": 82, "style": "possession"}
}

# ================= INITIALISATION DES DONNÉES =================
if "match_history" not in st.session_state:
    # Charge depuis fichier, sinon données par défaut
    saved_matches = load_matches_from_file()
    if saved_matches:
        st.session_state.match_history = saved_matches
        st.session_state.data_source = "fichier_sauvegardé"
    else:
        st.session_state.match_history = [
            "Real Madrid 3-1 Barcelona",
            "Manchester City 2-2 Liverpool", 
            "Bayern Munich 4-2 Dortmund",
            "PSG 2-0 Marseille",
            "Milan 1-0 Inter",
            "Arsenal 3-1 Tottenham",
            "Atletico Madrid 2-0 Real Madrid",
            "Juventus 1-0 Roma",
            "Manchester United 1-1 Chelsea",
            "Newcastle 4-1 Paris Saint Germain"
        ]
        st.session_state.data_source = "données_par_défaut"

# ================= FONCTIONS DU BOT =================
def parse_match(match_text):
    """Parse un match au format 'Équipe1 X-Y Équipe2'"""
    try:
        pattern = r"(.+?)\s+(\d+)-(\d+)\s+(.+)"
        match = re.match(pattern, match_text)
        if match:
            home, home_goals, away_goals, away = match.groups()
            return {
                'home': home.strip(),
                'away': away.strip(),
                'score': f"{home_goals}-{away_goals}",
                'home_goals': int(home_goals),
                'away_goals': int(away_goals),
                'text': match_text
            }
    except:
        pass
    return None

def calculate_team_stats():
    """Calcule les statistiques basées sur l'historique"""
    stats = defaultdict(lambda: {
        'scored': 0, 'conceded': 0, 'games': 0,
        'home_scored': 0, 'home_conceded': 0, 'home_games': 0,
        'away_scored': 0, 'away_conceded': 0, 'away_games': 0
    })
    
    for match_text in st.session_state.match_history:
        match = parse_match(match_text)
        if match:
            home = match['home']
            away = match['away']
            hg = match['home_goals']
            ag = match['away_goals']
            
            # Statistiques domicile
            stats[home]['scored'] += hg
            stats[home]['conceded'] += ag
            stats[home]['games'] += 1
            stats[home]['home_scored'] += hg
            stats[home]['home_conceded'] += ag
            stats[home]['home_games'] += 1
            
            # Statistiques extérieur
            stats[away]['scored'] += ag
            stats[away]['conceded'] += hg
            stats[away]['games'] += 1
            stats[away]['away_scored'] += ag
            stats[away]['away_conceded'] += hg
            stats[away]['away_games'] += 1
    
    return stats

def enhanced_poisson_prediction(home_team, away_team):
    """Prédiction Poisson améliorée avec facteurs multiples"""
    # Récupérer les stats FIFA
    home_fifa = FIFA_TEAMS.get(home_team, {'attack': 75, 'defense': 75, 'overall': 75})
    away_fifa = FIFA_TEAMS.get(away_team, {'attack': 75, 'defense': 75, 'overall': 75})
    
    # Calculer les stats historiques
    team_stats = calculate_team_stats()
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # Attaque moyenne (combine FIFA et historique)
    if home_stats['games'] > 0:
        home_attack_hist = home_stats['scored'] / home_stats['games'] * 10
    else:
        home_attack_hist = 75
    
    if away_stats['games'] > 0:
        away_attack_hist = away_stats['scored'] / away_stats['games'] * 10
    else:
        away_attack_hist = 75
    
    home_attack = (home_fifa['attack'] * 0.7 + home_attack_hist * 0.3)
    away_attack = (away_fifa['attack'] * 0.7 + away_attack_hist * 0.3)
    
    # Défense moyenne
    if home_stats['games'] > 0:
        home_defense_hist = 100 - (home_stats['conceded'] / home_stats['games'] * 10)
    else:
        home_defense_hist = 75
    
    if away_stats['games'] > 0:
        away_defense_hist = 100 - (away_stats['conceded'] / away_stats['games'] * 10)
    else:
        away_defense_hist = 75
    
    home_defense = (home_fifa['defense'] * 0.7 + home_defense_hist * 0.3)
    away_defense = (away_fifa['defense'] * 0.7 + away_defense_hist * 0.3)
    
    # Facteurs de calcul
    home_advantage = 1.15  # +15% à domicile
    form_factor = np.random.uniform(0.9, 1.1)  # Forme aléatoire
    
    # Différence de niveau
    level_diff = (home_fifa.get('overall', 75) - away_fifa.get('overall', 75)) / 100
    
    # Calcul final lambda
    lambda_home = max(0.3, (home_attack + (100 - away_defense)) / 100 * 1.8 * home_advantage * form_factor * (1 + level_diff))
    lambda_away = max(0.3, (away_attack + (100 - home_defense)) / 100 * 1.8 * 0.9 * form_factor * (1 - level_diff))
    
    return lambda_home, lambda_away

def simulate_match(home_team, away_team, simulations=10000):
    """Simulation Monte Carlo pour obtenir les scores probables"""
    lambda_home, lambda_away = enhanced_poisson_prediction(home_team, away_team)
    
    home_wins = draw = away_wins = 0
    score_counter = Counter()
    
    for _ in range(simulations):
        # Génération Poisson avec variation
        home_goals = int(np.random.poisson(lambda_home) + np.random.uniform(-0.3, 0.3))
        away_goals = int(np.random.poisson(lambda_away) + np.random.uniform(-0.3, 0.3))
        
        # Limiter à des scores réalistes
        home_goals = max(0, min(home_goals, 7))
        away_goals = max(0, min(away_goals, 7))
        
        score = f"{home_goals}-{away_goals}"
        score_counter[score] += 1
        
        if home_goals > away_goals:
            home_wins += 1
        elif home_goals == away_goals:
            draw += 1
        else:
            away_wins += 1
    
    total = simulations
    score_probabilities = {score: count/total for score, count in score_counter.most_common(20)}
    
    # TOP 5 scores les plus probables
    top_5_scores = [
        {"score": score, "probability": round(prob * 100, 2)}
        for score, prob in list(score_probabilities.items())[:5]
    ]
    
    return {
        'home_win_prob': home_wins / total,
        'draw_prob': draw / total,
        'away_win_prob': away_wins / total,
        'score_probabilities': score_probabilities,
        'top_5_scores': top_5_scores,
        'expected_goals': {'home': round(lambda_home, 2), 'away': round(lambda_away, 2)},
        'total_simulations': simulations
    }

def calculate_confidence(home_team, away_team, prediction):
    """Calcule la fiabilité de la prédiction (0-100%)"""
    confidence = 70  # Base
    
    # Vérifier les données historiques
    team_stats = calculate_team_stats()
    home_games = team_stats[home_team]['games']
    away_games = team_stats[away_team]['games']
    
    if home_games > 2 and away_games > 2:
        confidence += 15
    elif home_games > 0 or away_games > 0:
        confidence += 5
    
    # Différence de niveau FIFA
    home_fifa = FIFA_TEAMS.get(home_team, {'overall': 75})
    away_fifa = FIFA_TEAMS.get(away_team, {'overall': 75})
    diff = abs(home_fifa['overall'] - away_fifa['overall'])
    
    if diff > 10:
        confidence += 10
    if diff > 20:
        confidence += 5
    
    # Probabilité claire
    max_prob = max(prediction['home_win_prob'], prediction['draw_prob'], prediction['away_win_prob'])
    if max_prob > 0.5:
        confidence += (max_prob - 0.5) * 40
    
    # Nombre de simulations
    confidence = min(confidence + (prediction.get('total_simulations', 10000) / 100000 * 10), 95)
    
    return round(confidence)

# ================= INTERFACE STREAMLIT =================
st.title("⚽ FIFA PREDICTOR PRO")
st.markdown("**Prédictions intelligentes basées sur Poisson + Machine Learning**")

# SIDEBAR - Gestion des données
with st.sidebar:
    st.header("📊 Gestion des Données")
    
    # Affichage source données
    if hasattr(st.session_state, 'data_source'):
        if st.session_state.data_source == "fichier_sauvegardé":
            st.success("✓ Données chargées depuis sauvegarde")
        else:
            st.info("ℹ️ Données par défaut")
    
    # Ajouter un match
    st.subheader("➕ Ajouter un match")
    new_match = st.text_input(
        "Format: Équipe1 X-Y Équipe2",
        placeholder="Ex: Real Madrid 3-1 Barcelona",
        key="new_match_input",
        help="Saisissez exactement: NomÉquipe score score NomÉquipe"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Ajouter", use_container_width=True, type="primary"):
            if new_match:
                parsed = parse_match(new_match)
                if parsed:
                    if parsed['home'] in FIFA_TEAMS and parsed['away'] in FIFA_TEAMS:
                        st.session_state.match_history.append(new_match)
                        save_matches_to_file(st.session_state.match_history)
                        st.session_state.data_source = "fichier_sauvegardé"
                        st.success(f"✓ {new_match}")
                        st.rerun()
                    else:
                        st.error("Équipe non reconnue")
                else:
                    st.error("Format invalide. Ex: Real Madrid 2-1 Barcelona")
    
    with col2:
        if st.button("🗑️ Effacer", use_container_width=True, type="secondary"):
            st.session_state.match_history = []
            save_matches_to_file([])
            st.success("Historique effacé")
            st.rerun()
    
    st.divider()
    
    # Historique rapide
    st.subheader("📜 Derniers matchs")
    if st.session_state.match_history:
        for match in st.session_state.match_history[-5:]:
            st.text(f"• {match}")
        st.caption(f"Total: {len(st.session_state.match_history)} matchs")
    else:
        st.info("Aucun match enregistré")
    
    # Export/Import
    st.divider()
    st.subheader("💾 Sauvegarde")
    
    # Export
    if st.session_state.match_history:
        matches_text = "\n".join(st.session_state.match_history)
        st.download_button(
            label="📥 Exporter matchs (.txt)",
            data=matches_text,
            file_name="matchs_fifa.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Import
    uploaded_file = st.file_uploader("📤 Importer matchs", type=['txt'], key="file_uploader")
    if uploaded_file is not None:
        try:
            new_matches = uploaded_file.read().decode("utf-8").splitlines()
            valid_matches = []
            for m in new_matches:
                m = m.strip()
                if m and parse_match(m):
                    valid_matches.append(m)
            
            if valid_matches:
                st.session_state.match_history.extend(valid_matches)
                save_matches_to_file(st.session_state.match_history)
                st.success(f"{len(valid_matches)} matchs importés")
                st.rerun()
        except:
            st.error("Erreur lecture fichier")

# MAIN CONTENT - Onglets
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prédiction", "📈 Analyses", "🏆 Équipes", "📊 Statistiques"])

with tab1:
    st.header("🔮 Prédire un match")
    
    # Sélection équipes
    all_teams = sorted(list(FIFA_TEAMS.keys()))
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox(
            "🏠 Équipe domicile",
            all_teams,
            index=all_teams.index("Real Madrid") if "Real Madrid" in all_teams else 0,
            key="home_select"
        )
    
    with col2:
        # Filtrer l'équipe away
        away_options = [team for team in all_teams if team != home_team]
        away_team = st.selectbox(
            "✈️ Équipe extérieur",
            away_options,
            index=away_options.index("Barcelona") if "Barcelona" in away_options else 0,
            key="away_select"
        )
    
    # Paramètres simulation
    with st.expander("⚙️ Paramètres avancés"):
        sim_slider = st.slider("Nombre de simulations", 1000, 50000, 10000, 1000,
                             help="Plus de simulations = plus précis mais plus lent")
    
    # Bouton prédiction
    if st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True):
        with st.spinner(f"Simulation de {sim_slider} matchs..."):
            result = simulate_match(home_team, away_team, simulations=sim_slider)
            confidence = calculate_confidence(home_team, away_team, result)
        
        st.success("✅ Prédiction terminée")
        
        # Métriques résultats
        st.subheader("📊 Probabilités des résultats")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"Victoire {home_team}",
                f"{result['home_win_prob']*100:.1f}%",
                delta="Domicile"
            )
        
        with col2:
            st.metric(
                "Match nul",
                f"{result['draw_prob']*100:.1f}%",
                delta="Égalité"
            )
        
        with col3:
            st.metric(
                f"Victoire {away_team}",
                f"{result['away_win_prob']*100:.1f}%",
                delta="Extérieur"
            )
        
        # Barre de confiance
        st.progress(confidence/100, text=f"📈 Fiabilité de la prédiction: {confidence}%")
        
        st.divider()
        
        # TOP 5 SCORES
        st.subheader("🎯 TOP 5 Scores les plus probables")
        
        scores_cols = st.columns(5)
        score_data = result['top_5_scores']
        
        colors = ['#FF6B6B', '#FFA726', '#66BB6A', '#42A5F5', '#AB47BC']
        
        for idx in range(5):
            with scores_cols[idx]:
                if idx < len(score_data):
                    score = score_data[idx]
                    # Container stylé
                    st.markdown(f"""
                    <div style='
                        background: {colors[idx]};
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        color: white;
                        margin: 5px;
                    '>
                        <h3 style='margin:0; font-size: 24px;'>{score['score']}</h3>
                        <h2 style='margin:5px 0; font-size: 32px;'>{score['probability']}%</h2>
                        <p style='margin:0; font-size: 14px;'>#{idx+1}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Donnée indisponible")
        
        st.divider()
        
        # Graphique probabilités
        st.subheader("📈 Visualisation des résultats")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Victoire domicile', 'Match nul', 'Victoire extérieur'],
                y=[result['home_win_prob']*100, result['draw_prob']*100, result['away_win_prob']*100],
                marker_color=['#4CAF50', '#FF9800', '#F44336'],
                text=[f"{result['home_win_prob']*100:.1f}%", 
                      f"{result['draw_prob']*100:.1f}%", 
                      f"{result['away_win_prob']*100:.1f}%"],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            yaxis_title="Probabilité (%)",
            xaxis_title="Résultat possible",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Buts attendus
        st.subheader("⚽ Buts attendus (xG)")
        
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.metric(
                f"{home_team} (domicile)",
                result['expected_goals']['home'],
                help="Nombre moyen de buts attendu"
            )
        
        with ecol2:
            st.metric(
                f"{away_team} (extérieur)",
                result['expected_goals']['away'],
                help="Nombre moyen de buts attendu"
            )
        
        # Total buts
        total_goals = result['expected_goals']['home'] + result['expected_goals']['away']
        if total_goals > 3.0:
            st.info(f"🔴 Match à haut score attendu ({total_goals:.1f} buts total)")
        elif total_goals < 1.5:
            st.info(f"🔵 Match défensif attendu ({total_goals:.1f} buts total)")
        else:
            st.info(f"🟡 Score équilibré attendu ({total_goals:.1f} buts total)")

with tab2:
    st.header("📈 Analyses détaillées")
    
    if 'result' in locals():
        # Analyse tactique
        st.subheader("🎭 Analyse tactique du match")
        
        home_fifa = FIFA_TEAMS.get(home_team, {})
        away_fifa = FIFA_TEAMS.get(away_team, {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"##### 🏠 {home_team}")
            st.metric("Style de jeu", home_fifa.get('style', 'Non défini'))
            
            # Graphique radar attaque/défense
            categories = ['Attaque', 'Défense', 'Milieu']
            values_home = [
                home_fifa.get('attack', 75),
                home_fifa.get('defense', 75),
                home_fifa.get('midfield', 75)
            ]
            
            fig_home = go.Figure(data=go.Scatterpolar(
                r=values_home + values_home[:1],
                theta=categories + categories[:1],
                fill='toself',
                name=home_team,
                line=dict(color='#4CAF50')
            ))
            
            fig_home.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig_home, use_container_width=True)
        
        with col2:
            st.markdown(f"##### ✈️ {away_team}")
            st.metric("Style de jeu", away_fifa.get('style', 'Non défini'))
            
            # Graphique radar away
            values_away = [
                away_fifa.get('attack', 75),
                away_fifa.get('defense', 75),
                away_fifa.get('midfield', 75)
            ]
            
            fig_away = go.Figure(data=go.Scatterpolar(
                r=values_away + values_away[:1],
                theta=categories + categories[:1],
                fill='toself',
                name=away_team,
                line=dict(color='#F44336')
            ))
            
            fig_away.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig_away, use_container_width=True)
        
        st.divider()
        
        # Insights IA
        st.subheader("🤖 Insights & Recommandations")
        
        insights = []
        
        # Analyse résultat
        if result['home_win_prob'] > 0.6:
            insights.append(f"• **{home_team} est grand favori** à domicile ({(result['home_win_prob']*100):.1f}% de chances)")
        elif result['away_win_prob'] > 0.6:
            insights.append(f"• **{away_team} pourrait surprendre** à l'extérieur ({(result['away_win_prob']*100):.1f}% de chances)")
        
        if result['draw_prob'] > 0.35:
            insights.append(f"• **Match très serré**, nul à envisager ({(result['draw_prob']*100):.1f}% de chances)")
        
        # Analyse buts
        total_exp_goals = result['expected_goals']['home'] + result['expected_goals']['away']
        if total_exp_goals > 3.0:
            insights.append(f"• **Rencontre offensive** attendue ({total_exp_goals:.1f} buts au total)")
        elif total_exp_goals < 1.5:
            insights.append(f"• **Match tactique** avec peu de buts ({total_exp_goals:.1f} buts au total)")
        
        # Analyse style
        home_style = home_fifa.get('style', '')
        away_style = away_fifa.get('style', '')
        
        if home_style == 'counter_attack' and away_style == 'possession':
            insights.append(f"• **Duel tactique**: {home_team} en contre-attaque vs {away_team} en possession")
        elif home_style == 'defensive' and away_style == 'attacking':
            insights.append(f"• **Défense vs Attaque**: {home_team} solide défensivement face à {away_team} offensif")
        
        # Afficher insights
        for insight in insights:
            st.info(insight)
        
        # Distribution scores
        st.divider()
        st.subheader("📊 Distribution des scores simulés")
        
        if result['score_probabilities']:
            scores_df = pd.DataFrame(
                list(result['score_probabilities'].items())[:15],
                columns=['Score', 'Probabilité']
            )
            scores_df['Probabilité %'] = scores_df['Probabilité'] * 100
            
            fig_scores = go.Figure(data=[
                go.Bar(
                    x=scores_df['Score'],
                    y=scores_df['Probabilité %'],
                    marker_color='#2196F3',
                    text=scores_df['Probabilité %'].round(1).astype(str) + '%',
                    textposition='auto'
                )
            ])
            
            fig_scores.update_layout(
                xaxis_title="Score",
                yaxis_title="Probabilité (%)",
                height=400
            )
            
            st.plotly_chart(fig_scores, use_container_width=True)

with tab3:
    st.header("🏆 Équipes disponibles")
    
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        filter_league = st.selectbox(
            "Filtrer par championnat",
            ["Tous", "La Liga", "Premier League", "Bundesliga", "Serie A", "Autres"]
        )
    
    with col2:
        sort_by = st.selectbox(
            "Trier par",
            ["Note FIFA", "Attaque", "Défense", "Nom"]
        )
    
    # Préparation données
    teams_list = []
    for team, stats in FIFA_TEAMS.items():
        # Déterminer championnat
        league = "Autres"
        if team in ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia", "Villarreal"]:
            league = "La Liga"
        elif team in ["Manchester City", "Liverpool", "Arsenal", "Chelsea", "Manchester United", "Tottenham", "Newcastle", "Aston Villa"]:
            league = "Premier League"
        elif team in ["Bayern Munich", "Dortmund", "Leipzig", "Leverkusen"]:
            league = "Bundesliga"
        elif team in ["Juventus", "Milan", "Inter", "Napoli", "Roma"]:
            league = "Serie A"
        
        if filter_league == "Tous" or league == filter_league:
            teams_list.append({
                'Équipe': team,
                'Championnat': league,
                'Attaque': stats['attack'],
                'Défense': stats['defense'],
                'Milieu': stats.get('midfield', 75),
                'Note FIFA': stats['overall'],
                'Style': stats['style']
            })
    
    # Tri
    if sort_by == "Note FIFA":
        teams_list.sort(key=lambda x: x['Note FIFA'], reverse=True)
    elif sort_by == "Attaque":
        teams_list.sort(key=lambda x: x['Attaque'], reverse=True)
    elif sort_by == "Défense":
        teams_list.sort(key=lambda x: x['Défense'], reverse=True)
    elif sort_by == "Nom":
        teams_list.sort(key=lambda x: x['Équipe'])
    
    # Affichage
    df_teams = pd.DataFrame(teams_list)
    
    # Format personnalisé
    st.dataframe(
        df_teams,
        use_container_width=True,
        column_config={
            "Équipe": st.column_config.TextColumn(width="large"),
            "Championnat": st.column_config.TextColumn(width="medium"),
            "Attaque": st.column_config.ProgressColumn(
                format="%d",
                min_value=0,
                max_value=100,
                width="medium"
            ),
            "Défense": st.column_config.ProgressColumn(
                format="%d",
                min_value=0,
                max_value=100,
                width="medium"
            ),
            "Note FIFA": st.column_config.NumberColumn(
                format="%d",
                help="Note globale FIFA"
            ),
            "Style": st.column_config.TextColumn(width="medium")
        },
        hide_index=True
    )
    
    # Statistiques globales
    st.subheader("📈 Statistiques globales des équipes")
    
    if len(teams_list) > 0:
        avg_attack = df_teams['Attaque'].mean()
        avg_defense = df_teams['Défense'].mean()
        avg_overall = df_teams['Note FIFA'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Attaque moyenne", f"{avg_attack:.1f}")
        with col2:
            st.metric("Défense moyenne", f"{avg_defense:.1f}")
        with col3:
            st.metric("Note moyenne", f"{avg_overall:.1f}")

with tab4:
    st.header("📊 Statistiques historiques")
    
    team_stats = calculate_team_stats()
    
    # Top 5 attaque
    st.subheader("🥇 Top 5 - Attaque")
    attack_ranking = []
    for team, stats in team_stats.items():
        if stats['games'] > 0:
            avg_scored = stats['scored'] / stats['games']
            attack_ranking.append((team, avg_scored, stats['scored'], stats['games']))
    
    attack_ranking.sort(key=lambda x: x[1], reverse=True)
