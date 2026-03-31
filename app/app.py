import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Loading model and required encoders...")

    model                  = joblib.load(os.path.join(BASE_DIR, 'models', 'model',      'best_model.pkl'))
    encode                 = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'encode.pkl'))
    label_encoders         = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'label_encoders.pkl'))
    feature_columns        = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'feature_columns.pkl'))
    team_categories        = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'team_categories.pkl'))

    team_name_mapping      = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings',   'team_name_mapping.pkl'))
    venue_mapping          = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings',   'venue_mapping.pkl'))
    top_venues             = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings',   'top_venues.pkl'))
    high_toss_venues       = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings',   'high_toss_impact_venues.pkl'))

    team_win_rate          = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'team_win_rate.pkl'))
    context_wins           = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'context_wins.pkl'))
    context_matches        = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'context_matches.pkl'))
    h2h_wins               = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'h2h_wins.pkl'))
    h2h_total              = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'h2h_total.pkl'))
    venue_wins             = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'venue_wins.pkl'))
    venue_total            = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'venue_total.pkl'))
    current_season_form    = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'current_season_form.pkl'))

    # Reverse mapping: encoded int → team name
    reverse_encode = {v: k for k, v in encode.items()}

    print("All artifacts loaded successfully!")

except Exception as e:
    print("Error loading models:", e)


def get_h2h_win_rate(team1: str, team2: str) -> float:
    """Head-to-head win rate for team1 against team2."""
    key = (team1, team2)
    if key in h2h_total.index and team1 in h2h_wins.columns:
        wins  = h2h_wins.loc[key, team1] if key in h2h_wins.index else 0
        total = h2h_total[key]
        return wins / total if total > 0 else 0.5
    return 0.5


def get_venue_win_rate(team: str, venue: str) -> float:
    """Win rate for a team at a specific venue."""
    if venue in venue_total.index and team in venue_wins.columns:
        wins  = venue_wins.loc[venue, team] if venue in venue_wins.index else 0
        total = venue_total[venue]
        return wins / total if total > 0 else 0.5
    return 0.5


def get_context_win_prob(team: str, ctx: str) -> float:
    """Historical win probability for team1 in this exact match context."""
    try:
        if ctx in context_wins.index and team in context_wins.columns:
            return context_wins.loc[ctx, team] / context_matches[ctx]
    except KeyError:
        pass
    return 0.5


def get_team_form(team: str) -> float:
    """
    Current season form for the team.
    Falls back to historical win rate if team not in current season data.
    """
    if team in current_season_form.index:
        return float(current_season_form[team])
    return float(team_win_rate.get(team, 0.5))


@app.route('/')
def home():
    try:
        valid_teams  = sorted(team_categories)
        valid_venues = sorted(list(top_venues))
        if 'Other' not in valid_venues:
            valid_venues.append('Other')
        return render_template('index.html', teams=valid_teams, venues=valid_venues)
    except Exception as e:
        return f"Model not loaded yet. Error: {e}"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data         = request.json
        team1        = data['team1']
        team2        = data['team2']
        toss_winner  = data['toss_winner']
        toss_decision= data['toss_decision']
        venue        = data['venue']

        # ── Normalise names ──────────────────────────────────────────────
        team1       = team_name_mapping.get(team1, team1)
        team2       = team_name_mapping.get(team2, team2)
        toss_winner = team_name_mapping.get(toss_winner, toss_winner)
        venue       = venue_mapping.get(venue, venue)
        if venue not in top_venues:
            venue = 'Other'

        # ── Basic toss features ──────────────────────────────────────────
        toss_decision_encoded = 0 if toss_decision.lower() == 'bat' else 1
        toss_win_team1        = 1 if toss_winner == team1 else 0
        team1_batting_first   = int(
            (toss_winner == team1 and toss_decision_encoded == 0) or
            (toss_winner == team2 and toss_decision_encoded == 1)
        )

        # Toss-venue interaction — only meaningful at high-dew venues
        toss_matters             = 1 if venue in high_toss_venues else 0
        effective_toss_advantage = toss_win_team1 * toss_matters

        # ── Team encoding ────────────────────────────────────────────────
        t1_enc = encode.get(team1, -1)
        t2_enc = encode.get(team2, -1)
        tw_enc = encode.get(toss_winner, -1)

        if -1 in (t1_enc, t2_enc, tw_enc):
            unknown = [n for n, v in zip(
                ['team1', 'team2', 'toss_winner'],
                [t1_enc, t2_enc, tw_enc]
            ) if v == -1]
            return jsonify({'success': False, 'error': f"Unknown team(s): {unknown}"})

        # ── Venue encoding ───────────────────────────────────────────────
        v_enc = label_encoders['venue'].transform([venue])[0]

        # ── Stat features ────────────────────────────────────────────────
        strength_diff = (
            float(team_win_rate.get(team1, 0)) -
            float(team_win_rate.get(team2, 0))
        )

        h2h_win_rate_team1   = get_h2h_win_rate(team1, team2)
        # h2h_win_rate_team2 dropped (= 1 - team1, redundant)

        team1_venue_win_rate = get_venue_win_rate(team1, venue)
        team2_venue_win_rate = get_venue_win_rate(team2, venue)

        ctx = f"{team1}_{team2}_{venue}_{toss_winner}"
        team1_context_win_prob = get_context_win_prob(team1, ctx)
        # team2_context_win_prob dropped (near-mirror, removed from model)

        # ── Current season form (live external data) ─────────────────────
        team1_form = get_team_form(team1)
        team2_form = get_team_form(team2)

        # ── Build input DataFrame ────────────────────────────────────────
        input_dict = {
            'team1':                   t1_enc,
            'team2':                   t2_enc,
            'toss_winner':             tw_enc,
            'toss_decision':           toss_decision_encoded,
            'venue':                   v_enc,
            'toss_win_team1':          toss_win_team1,
            'team1_batting_first':     team1_batting_first,
            'toss_matters':            toss_matters,
            'effective_toss_advantage':effective_toss_advantage,
            'strength_diff':           strength_diff,
            'h2h_win_rate_team1':      h2h_win_rate_team1,
            'team1_venue_win_rate':    team1_venue_win_rate,
            'team2_venue_win_rate':    team2_venue_win_rate,
            'team1_context_win_prob':  team1_context_win_prob,
            'team1_form':              team1_form,
            'team2_form':              team2_form,
        }

        input_df = pd.DataFrame([input_dict])

        # Align columns to exactly what the model was trained on
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # ── Predict ──────────────────────────────────────────────────────
        pred_encoded     = model.predict(input_df)[0]
        pred_proba       = model.predict_proba(input_df)[0]
        predicted_winner = reverse_encode.get(pred_encoded, "Unknown Team")

        # Win probabilities for both teams
        team1_win_prob = round(float(pred_proba[t1_enc]) * 100, 1)
        team2_win_prob = round(float(pred_proba[t2_enc]) * 100, 1)

        return jsonify({
            'success':    True,
            'winner':     predicted_winner,
            'team1_prob': team1_win_prob,
            'team2_prob': team2_win_prob,
            'team1':      team1,
            'team2':      team2,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)