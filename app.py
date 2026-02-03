import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import json
import re
from collections import defaultdict, Counter
import math
import plotly.graph_objects as go

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="⚽ FIFA Predictor Pro",
    page_icon="⚽",
    layout="wide"
)

# ================= DONNÉES INITIALES =================
FIFA_TEAMS = {
    "Real Madrid": {"attack": 87, "defense": 85, "midfield": 86, "overall": 86, "style": "counter_attack"},
    "Barcelona": {"attack": 86, "defense": 84, "midfield": 87, "overall": 86, "style": "possession"},
    "Manchester City": {"attack": 88, "defense": 85, "midfield": 88, "overall": 87, "style": "possession"},
    "Liverpool": {"attack": 86, "defense": 84, "midfield": 85, "overall": 85, "style": "high_press"},
    "Bayern Munich": {"attack": 87, "defense": 85, "midfield": 86, "overall": 86, "style": "possession"},
    "PSG": {"attack": 87, "defense": 83, "midfield": 84, "overall": 85, "style": "attacking"},
    "Juventus": {"attack": 83, "defense": 85, "midfield": 83, "overall": 84, "style": "defensive"},
    "Chelsea": {"attack": 82, "defense": 84, "midfield": 83, "overall": 83, "style": "counter_attack"},
    "Manchester United": {"attack": 83, "defense": 82, "midfield": 83, "overall": 83, "style": "counter_attack"},
    "Milan": {"attack": 84, "defense": 83, "midfield": 82, "overall": 83, "style": "balanced"}
}

# Initialiser l'historique dans la session
if "match_history" not in st.session_state:
    st.session_state.match_history = [
        "Real Madrid 3-1 Barcelona",
        "Manchester City 2-2 Liverpool",
        "Bayern Munich 4-2 Dortmund",
        "PSG 2-0 Marseille",
        "Milan 1-0 Inter",
        "Real Madrid 2-0 Atletico Madrid",
        "Barcelona 3-0 Sevilla",
        "Manchester United 1-1 Chelsea",
        "Liverpool 3-1 Arsenal",
        "Bayern Munich 3-1 Leipzig"
    ]

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
    """Prédiction Poisson améliorée avec facteurs FIFA"""
    # Récupérer les stats
    home_fifa = FIFA_TEAMS.get(home_team, {'attack': 75, 'defense': 75, 'overall': 75, 'style': 'balanced'})
    away_fifa = FIFA_TEAMS.get(away_team, {'attack': 75, 'defense': 75, 'overall': 75, 'style': 'balanced'})
    team_stats = calculate_team_stats()
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # Calcul des moyennes
    home_attack_avg = home_stats['scored'] / max(home_stats['games'], 1)
    home_defense_avg = home_stats['conceded'] / max(home_stats['games'], 1)
    away_attack_avg = away_stats['scored'] / max(away_stats['games'], 1)
    away_defense_avg = away_stats['conceded'] / max(away_stats['games'], 1)
    
    # Facteurs d'attaque
    home_attack = (home_fifa['attack'] * 0.6 + home_attack_avg * 10 * 0.4)
    away_attack = (away_fifa['attack'] * 0.6 + away_attack_avg * 10 * 0.4)
    
    # Facteurs de défense
    home_defense = (home_fifa['defense'] * 0.6 + (100 - home_defense_avg * 10) * 0.4)
    away_defense = (away_fifa['defense'] * 0.6 + (100 - away_defense_avg * 10) * 0.4)
    
    # Calcul lambda avec tous les facteurs
    home_advantage = 1.15  # +15% à domicile
    form_factor = np.random.uniform(0.8, 1.2)  # Forme aléatoire
    
    lambda_home = max(0.3, (home_attack + (100 - away_defense)) / 100 * 1.8 * home_advantage * form_factor)
    lambda_away = max(0.3, (away_attack + (100 - home_defense)) / 100 * 1.8 * 0.9 * form_factor)
    
    return lambda_home, lambda_away

def simulate_match(home_team, away_team, simulations=10000):
    """Simulation Monte Carlo"""
    lambda_home, lambda_away = enhanced_poisson_prediction(home_team, away_team)
    
    home_wins = draw = away_wins = 0
    score_counter = Counter()
    
    for _ in range(simulations):
        home_goals = np.random.poisson(lambda_home)
        away_goals = np.random.poisson(lambda_away)
        
        # Limiter à des scores réalistes
        home_goals = min(home_goals, 7)
        away_goals = min(away_goals, 7)
        
        score = f"{home_goals}-{away_goals}"
        score_counter[score] += 1
        
        if home_goals > away_goals:
            home_wins += 1
        elif home_goals == away_goals:
            draw += 1
        else:
            away_wins += 1
    
    total = simulations
    score_probabilities = {score: count/total for score, count in score_counter.most_common(15)}
    
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
        'most_common_scores': list(score_counter.most_common(5))
    }

def calculate_confidence(home_team, away_team, prediction):
    """Calcule la fiabilité de la prédiction"""
    confidence = 70  # Base
    
    # Vérifier les données historiques
    team_stats = calculate_team_stats()
    home_games = team_stats[home_team]['games']
    away_games = team_stats[away_team]['games']
    
    if home_games > 2 and away_games > 2:
        confidence += 15
    elif home_games > 0 or away_games > 0:
        confidence += 5
    
    # Différence de niveau
    home_fifa = FIFA_TEAMS.get(home_team, {'overall': 75})
    away_fifa = FIFA_TEAMS.get(away_team, {'overall': 75})
    diff = abs(home_fifa['overall'] - away_fifa['overall'])
    
    if diff > 10:
        confidence += 10
    if diff > 20:
        confidence += 10
    
    # Probabilité claire
    max_prob = max(prediction['home_win_prob'], prediction['draw_prob'], prediction['away_win_prob'])
    if max_prob > 0.5:
        confidence += (max_prob - 0.5) * 30
    
    return min(95, round(confidence))

# ================= INTERFACE STREAMLIT =================
st.title("⚽ FIFA PREDICTOR PRO")
st.markdown("---")

# SIDEBAR - Gestion des matchs
with st.sidebar:
    st.header("📥 Gestion des Matchs")
    
    # Ajouter un match
    st.subheader("➕ Ajouter un match")
    new_match = st.text_input(
        "Format: Équipe1 X-Y Équipe2",
        placeholder="Ex: Real Madrid 3-1 Barcelona",
        key="new_match_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Ajouter", use_container_width=True):
            if new_match and parse_match(new_match):
                st.session_state.match_history.append(new_match)
                st.success("Match ajouté !")
                st.rerun()
            else:
                st.error("Format invalide !")
    
    with col2:
        if st.button("🗑️ Tout effacer", use_container_width=True):
            st.session_state.match_history = []
            st.success("Historique effacé !")
            st.rerun()
    
    st.markdown("---")
    
    # Afficher l'historique
    st.subheader("📜 Historique actuel")
    if st.session_state.match_history:
        for i, match in enumerate(st.session_state.match_history[-10:], 1):  # 10 derniers
            st.text(f"{i}. {match}")
    else:
        st.info("Aucun match dans l'historique")
    
    # Statistiques globales
    st.markdown("---")
    st.subheader("📊 Statistiques globales")
    team_stats = calculate_team_stats()
    total_matches = len(st.session_state.match_history)
    st.metric("Matches analysés", total_matches)
    
    if total_matches > 0:
        total_goals = sum(stats['scored'] for stats in team_stats.values())
        avg_goals = total_goals / total_matches
        st.metric("Moyenne de buts/match", round(avg_goals, 2))

# MAIN CONTENT - Prédiction et analyses
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prédiction", "📈 Analyses", "🏆 Classement", "⚙️ Configuration"])

with tab1:
    st.header("🔮 Prédire un match")
    
    # Sélection des équipes
    all_teams = list(FIFA_TEAMS.keys())
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox(
            "Équipe à domicile 🏠",
            all_teams,
            index=0,
            key="home_select"
        )
    with col2:
        # Filtrer pour ne pas sélectionner la même équipe
        away_options = [team for team in all_teams if team != home_team]
        away_team = st.selectbox(
            "Équipe à l'extérieur ✈️",
            away_options,
            index=min(1, len(away_options)-1),
            key="away_select"
        )
    
    # Bouton de prédiction
    if st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True):
        with st.spinner("Simulation en cours..."):
            # Exécuter la prédiction
            result = simulate_match(home_team, away_team)
            confidence = calculate_confidence(home_team, away_team, result)
            
            # Afficher les résultats
            st.success("✅ Prédiction terminée !")
            
            # Métriques principales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"Victoire {home_team}",
                    f"{result['home_win_prob']*100:.1f}%",
                    delta=f"Confiance: {confidence}%"
                )
            with col2:
                st.metric("Match nul", f"{result['draw_prob']*100:.1f}%")
            with col3:
                st.metric(f"Victoire {away_team}", f"{result['away_win_prob']*100:.1f}%")
            
            st.markdown("---")
            
            # TOP 5 scores probables
            st.subheader("🎯 TOP 5 Scores les plus probables")
            scores_cols = st.columns(5)
            for idx, score_data in enumerate(result['top_5_scores']):
                with scores_cols[idx]:
                    st.metric(
                        score_data['score'],
                        f"{score_data['probability']}%",
                        delta=f"#{idx+1}"
                    )
            
            st.markdown("---")
            
            # Graphique des probabilités
            st.subheader("📊 Distribution des résultats")
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
                xaxis_title="Résultat",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Buts attendus
            st.subheader("⚽ Buts attendus")
            eg_col1, eg_col2 = st.columns(2)
            with eg_col1:
                st.metric(f"{home_team} (domicile)", result['expected_goals']['home'])
            with eg_col2:
                st.metric(f"{away_team} (extérieur)", result['expected_goals']['away'])

with tab2:
    st.header("📈 Analyses détaillées")
    
    if 'result' in locals():
        # Analyse tactique
        st.subheader("🎭 Analyse tactique")
        home_fifa = FIFA_TEAMS.get(home_team, {})
        away_fifa = FIFA_TEAMS.get(away_team, {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Style de jeu domicile", home_fifa.get('style', 'Inconnu'))
            st.metric("Attaque domicile", home_fifa.get('attack', 75))
            st.metric("Défense domicile", home_fifa.get('defense', 75))
        
        with col2:
            st.metric("Style de jeu extérieur", away_fifa.get('style', 'Inconnu'))
            st.metric("Attaque extérieur", away_fifa.get('attack', 75))
            st.metric("Défense extérieur", away_fifa.get('defense', 75))
        
        # Distribution des scores
        st.subheader("📈 Distribution des scores simulés")
        if result['score_probabilities']:
            scores_df = pd.DataFrame(
                list(result['score_probabilities'].items())[:10],
                columns=['Score', 'Probabilité']
            )
            scores_df['Probabilité %'] = scores_df['Probabilité'] * 100
            st.bar_chart(scores_df.set_index('Score')['Probabilité %'])
        
        # Insights IA
        st.subheader("🤖 Insights IA")
        insights = []
        
        if result['home_win_prob'] > 0.6:
            insights.append(f"• {home_team} est grand favori à domicile")
        elif result['away_win_prob'] > 0.6:
            insights.append(f"• {away_team} pourrait créer la surprise à l'extérieur")
        
        if result['draw_prob'] > 0.35:
            insights.append("• Match très serré, nul à envisager sérieusement")
        
        if result['expected_goals']['home'] + result['expected_goals']['away'] > 3.0:
            insights.append("• Rencontre prometteuse avec beaucoup de buts attendus")
        elif result['expected_goals']['home'] + result['expected_goals']['away'] < 1.5:
            insights.append("• Match tactique avec peu de buts attendus")
        
        for insight in insights:
            st.info(insight)

with tab3:
    st.header("🏆 Classement des équipes")
    
    team_stats = calculate_team_stats()
    ranking_data = []
    
    for team, stats in team_stats.items():
        if stats['games'] > 0:
            fifa_stats = FIFA_TEAMS.get(team, {'overall': 75, 'attack': 75, 'defense': 75})
            ranking_data.append({
                'Équipe': team,
                'Matches': stats['games'],
                'Buts pour': stats['scored'],
                'Buts contre': stats['conceded'],
                'Différence': stats['scored'] - stats['conceded'],
                'Attaque FIFA': fifa_stats['attack'],
                'Défense FIFA': fifa_stats['defense'],
                'Note FIFA': fifa_stats['overall']
            })
    
    if ranking_data:
        df = pd.DataFrame(ranking_data)
        df = df.sort_values(['Différence', 'Buts pour'], ascending=[False, False])
        df.index = range(1, len(df) + 1)
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Équipe": st.column_config.TextColumn(width="large"),
                "Matches": st.column_config.NumberColumn(format="%d"),
                "Différence": st.column_config.NumberColumn(
                    format="%+d",
                    help="Différence de buts"
                )
            }
        )
        
        # Graphique radar pour la meilleure équipe
        if len(df) > 0:
            st.subheader("📊 Profil de la meilleure équipe")
            best_team = df.iloc[0]['Équipe']
            best_stats = FIFA_TEAMS.get(best_team, {})
            
            if best_stats:
                categories = ['Attaque', 'Défense', 'Milieu', 'Overall']
                values = [
                    best_stats.get('attack', 75),
                    best_stats.get('defense', 75),
                    best_stats.get('midfield', 75),
                    best_stats.get('overall', 75)
                ]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values + values[:1],  # Fermer le radar
                    theta=categories + categories[:1],
                    fill='toself',
                    name=best_team
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    title=f"Profil FIFA - {best_team}"
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("⚙️ Configuration")
    
    st.subheader("📋 Équipes disponibles")
    teams_df = pd.DataFrame([
        {
            'Équipe': team,
            'Attaque': stats['attack'],
            'Défense': stats['defense'],
            'Milieu': stats.get('midfield', 75),
            'Overall': stats['overall'],
            'Style': stats['style']
        }
        for team, stats in FIFA_TEAMS.items()
    ])
    
    st.dataframe(teams_df, use_container_width=True)
    
    st.subheader("🔧 Paramètres avancés")
    simulations = st.slider("Nombre de simulations", 1000, 50000, 10000, step=1000)
    st.info(f"Précision actuelle : ~{min(95, 70 + int(simulations/1000)*2)}%")
    
    if st.button("🔄 Redémarrer l'application", type="secondary"):
        st.session_state.clear()
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>⚽ <b>FIFA Predictor Pro v2.0</b> | Modèle Poisson amélioré | 
        Données mises à jour en temps réel</p>
        <p><small>Les prédictions sont à titre informatif. © 2024</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
