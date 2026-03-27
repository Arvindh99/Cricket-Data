import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models and mappings
try:
    print("Loading model and required encoders...")
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'model', 'best_model.pkl'))
    encode = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'encode.pkl'))
    label_encoders = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'label_encoders.pkl'))
    feature_columns = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'feature_columns.pkl'))
    team_name_mapping = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings', 'team_name_mapping.pkl'))
    venue_mapping = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings', 'venue_mapping.pkl'))
    top_venues = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings', 'top_venues.pkl'))
    team_win_rate = joblib.load(os.path.join(BASE_DIR, 'models', 'stats', 'team_win_rate.pkl'))
    context_wins = joblib.load(os.path.join(BASE_DIR, 'models', 'stats', 'context_wins.pkl'))
    context_matches = joblib.load(os.path.join(BASE_DIR, 'models', 'stats', 'context_matches.pkl'))
    print("All artifacts loaded successfully!")
except Exception as e:
    print("Error loading models:", e)

@app.route('/')
def home():
    try:
        valid_teams = sorted(list(encode['team1'].keys()))
        valid_venues = sorted(list(top_venues))
        if 'Other' not in valid_venues:
            valid_venues.append('Other')
        return render_template('index.html', teams=valid_teams, venues=valid_venues)
    except:
        return "Model not generated yet! Please wait for background training."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        team1 = data['team1']
        team2 = data['team2']
        toss_winner = data['toss_winner']
        toss_decision = data['toss_decision']
        venue = data['venue']

        team1 = team_name_mapping.get(team1, team1)
        team2 = team_name_mapping.get(team2, team2)
        toss_winner = team_name_mapping.get(toss_winner, toss_winner)
        venue = venue_mapping.get(venue, venue)
        
        if venue not in top_venues:
            venue = 'Other'
            
        toss_win_team1 = 1 if toss_winner == team1 else 0
        toss_decision_encoded = 0 if toss_decision.lower() == 'bat' else 1
        
        team1_batting_first = 1 if ((toss_winner == team1 and toss_decision_encoded == 0) or (toss_winner == team2 and toss_decision_encoded == 1)) else 0
        
        team1_strength = team_win_rate.get(team1, 0)
        team2_strength = team_win_rate.get(team2, 0)
        strength_diff = team1_strength - team2_strength
        
        t1_enc = encode.get('team1', {}).get(team1, -1)
        t2_enc = encode.get('team2', {}).get(team2, -1)
        tw_enc = encode.get('toss_winner', {}).get(toss_winner, -1)
        
        ctx = f"{team1}_{team2}_{venue}_{toss_winner}"
        try:
            t1_prob = context_wins.loc[ctx, team1] / context_matches[ctx] if team1 in context_wins.columns else 0.5
        except KeyError:
            t1_prob = 0.5
        try:
            t2_prob = context_wins.loc[ctx, team2] / context_matches[ctx] if team2 in context_wins.columns else 0.5
        except KeyError:
            t2_prob = 0.5
            
        v_enc = label_encoders['venue'].transform([venue])[0]
        
        input_data = pd.DataFrame([{
            'team1': t1_enc,
            'team2': t2_enc,
            'toss_winner': tw_enc,
            'toss_decision': toss_decision_encoded,
            'venue': v_enc,
            'toss_win_team1': toss_win_team1,
            'team1_batting_first': team1_batting_first,
            'strength_diff': strength_diff,
            'team1_context_win_prob': t1_prob,
            'team2_context_win_prob': t2_prob
        }])
        
        input_data = input_data[feature_columns]
        pred_encoded = model.predict(input_data)[0]
        
        reverse_encode = {v: k for k, v in encode['winner'].items()}
        predicted_winner = reverse_encode.get(pred_encoded, "Unknown Team")
        
        return jsonify({'success': True, 'winner': predicted_winner})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
