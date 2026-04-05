import os
import json
import joblib
import traceback
import pandas as pd
from datetime import datetime, timezone


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH      = os.path.join(BASE_DIR, 'data',   'csv',       'ipl_current_season.csv')
OUTPUT_PATH   = os.path.join(BASE_DIR, 'models', 'stats',     'season_eval.json')

def load_artifacts():
    print("Loading artifacts...")
    arts = {}
    arts['model']               = joblib.load(os.path.join(BASE_DIR, 'models', 'model',      'best_model.pkl'))
    arts['encode']              = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'encode.pkl'))
    arts['label_encoders']      = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'label_encoders.pkl'))
    arts['feature_columns']     = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocess', 'feature_columns.pkl'))
    arts['team_name_mapping']   = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings',   'team_name_mapping.pkl'))
    arts['venue_mapping']       = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings',   'venue_mapping.pkl'))
    arts['top_venues']          = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings',   'top_venues.pkl'))
    arts['high_toss_venues']    = joblib.load(os.path.join(BASE_DIR, 'models', 'mappings',   'high_toss_impact_venues.pkl'))
    arts['team_win_rate']       = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'team_win_rate.pkl'))
    arts['context_wins']        = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'context_wins.pkl'))
    arts['context_matches']     = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'context_matches.pkl'))
    arts['h2h_wins']            = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'h2h_wins.pkl'))
    arts['h2h_total']           = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'h2h_total.pkl'))
    arts['venue_wins']          = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'venue_wins.pkl'))
    arts['venue_total']         = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'venue_total.pkl'))
    arts['current_season_form'] = joblib.load(os.path.join(BASE_DIR, 'models', 'stats',      'current_season_form.pkl'))
    arts['reverse_encode']      = {v: k for k, v in arts['encode'].items()}
    print("  ✔ All artifacts loaded.")
    return arts

def get_h2h_win_rate(team1, team2, h2h_wins, h2h_total):
    key = (team1, team2)
    if key in h2h_total.index and team1 in h2h_wins.columns:
        wins  = h2h_wins.loc[key, team1] if key in h2h_wins.index else 0
        total = h2h_total[key]
        return wins / total if total > 0 else 0.5
    return 0.5


def get_venue_win_rate(team, venue, venue_wins, venue_total):
    if venue in venue_total.index and team in venue_wins.columns:
        wins  = venue_wins.loc[venue, team] if venue in venue_wins.index else 0
        total = venue_total[venue]
        return wins / total if total > 0 else 0.5
    return 0.5


def get_context_win_prob(team, ctx, context_wins, context_matches):
    try:
        if ctx in context_wins.index and team in context_wins.columns:
            return context_wins.loc[ctx, team] / context_matches[ctx]
    except KeyError:
        pass
    return 0.5


def get_team_form(team, current_season_form, team_win_rate):
    if team in current_season_form.index:
        return float(current_season_form[team])
    return float(team_win_rate.get(team, 0.5))

def build_features(row, arts):
    team1        = row['team1']
    team2        = row['team2']
    toss_winner  = row['toss_winner']
    toss_decision= str(row['toss_decision']).lower()
    venue        = row['venue']

    encode            = arts['encode']
    label_encoders    = arts['label_encoders']
    feature_columns   = arts['feature_columns']
    team_name_mapping = arts['team_name_mapping']
    venue_mapping     = arts['venue_mapping']
    top_venues        = arts['top_venues']
    high_toss_venues  = arts['high_toss_venues']
    team_win_rate     = arts['team_win_rate']
    context_wins      = arts['context_wins']
    context_matches   = arts['context_matches']
    h2h_wins          = arts['h2h_wins']
    h2h_total         = arts['h2h_total']
    venue_wins        = arts['venue_wins']
    venue_total       = arts['venue_total']
    current_season_form = arts['current_season_form']

    # Normalise
    team1       = team_name_mapping.get(team1, team1)
    team2       = team_name_mapping.get(team2, team2)
    toss_winner = team_name_mapping.get(toss_winner, toss_winner)
    venue       = venue_mapping.get(venue, venue)
    if venue not in top_venues:
        venue = 'Other'

    toss_decision_encoded = 0 if toss_decision == 'bat' else 1
    toss_win_team1        = 1 if toss_winner == team1 else 0
    team1_batting_first   = int(
        (toss_winner == team1 and toss_decision_encoded == 0) or
        (toss_winner == team2 and toss_decision_encoded == 1)
    )
    toss_matters             = 1 if venue in high_toss_venues else 0
    effective_toss_advantage = toss_win_team1 * toss_matters

    t1_enc = encode.get(team1, -1)
    t2_enc = encode.get(team2, -1)
    tw_enc = encode.get(toss_winner, -1)

    if -1 in (t1_enc, t2_enc, tw_enc):
        return None, None, None   # unknown team — skip

    v_enc = label_encoders['venue'].transform([venue])[0]

    strength_diff          = float(team_win_rate.get(team1, 0)) - float(team_win_rate.get(team2, 0))
    h2h_win_rate_team1     = get_h2h_win_rate(team1, team2, h2h_wins, h2h_total)
    team1_venue_win_rate   = get_venue_win_rate(team1, venue, venue_wins, venue_total)
    team2_venue_win_rate   = get_venue_win_rate(team2, venue, venue_wins, venue_total)
    ctx                    = f"{team1}_{team2}_{venue}_{toss_winner}"
    team1_context_win_prob = get_context_win_prob(team1, ctx, context_wins, context_matches)
    team1_form             = get_team_form(team1, current_season_form, team_win_rate)
    team2_form             = get_team_form(team2, current_season_form, team_win_rate)

    input_dict = {
        'team1':                    t1_enc,
        'team2':                    t2_enc,
        'toss_winner':              tw_enc,
        'toss_decision':            toss_decision_encoded,
        'venue':                    v_enc,
        'toss_win_team1':           toss_win_team1,
        'team1_batting_first':      team1_batting_first,
        'toss_matters':             toss_matters,
        'effective_toss_advantage': effective_toss_advantage,
        'strength_diff':            strength_diff,
        'h2h_win_rate_team1':       h2h_win_rate_team1,
        'team1_venue_win_rate':     team1_venue_win_rate,
        'team2_venue_win_rate':     team2_venue_win_rate,
        'team1_context_win_prob':   team1_context_win_prob,
        'team1_form':               team1_form,
        'team2_form':               team2_form,
    }

    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)
    return input_df, t1_enc, t2_enc

def evaluate():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Nothing to evaluate.")
        return

    arts = load_artifacts()

    df = pd.read_csv(CSV_PATH)

    # Normalise team/winner names
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        if col in df.columns:
            df[col] = df[col].replace(arts['team_name_mapping'])

    # Only process rows that have a result (completed matches)
    completed = df[df['winner'].notna() & (df['winner'].str.strip() != '')].copy()
    print(f"\nTotal matches in CSV  : {len(df)}")
    print(f"Completed matches     : {len(completed)}")

    results   = []
    correct   = 0
    skipped   = 0

    for _, row in completed.iterrows():
        try:
            input_df, t1_enc, t2_enc = build_features(row, arts)

            if input_df is None:
                skipped += 1
                continue

            pred_encoded     = arts['model'].predict(input_df)[0]
            pred_proba       = arts['model'].predict_proba(input_df)[0]
            predicted_winner = arts['reverse_encode'].get(pred_encoded, 'Unknown')

            # Normalise actual winner
            actual_winner    = arts['team_name_mapping'].get(row['winner'], row['winner'])

            is_correct       = predicted_winner == actual_winner
            if is_correct:
                correct += 1

            # Win probabilities
            team1_prob = round(float(pred_proba[t1_enc]) * 100, 1) if 0 <= t1_enc < len(pred_proba) else 50.0
            team2_prob = round(float(pred_proba[t2_enc]) * 100, 1) if 0 <= t2_enc < len(pred_proba) else 50.0

            results.append({
                'team1':            arts['team_name_mapping'].get(row['team1'], row['team1']),
                'team2':            arts['team_name_mapping'].get(row['team2'], row['team2']),
                'venue':            str(row.get('venue', '')),
                'toss_winner':      arts['team_name_mapping'].get(row.get('toss_winner', ''), row.get('toss_winner', '')),
                'toss_decision':    str(row.get('toss_decision', '')),
                'predicted_winner': predicted_winner,
                'actual_winner':    actual_winner,
                'team1_prob':       team1_prob,
                'team2_prob':       team2_prob,
                'correct':          is_correct,
            })

        except Exception as e:
            print(f"  ✘ Error on row {_}: {e}")
            traceback.print_exc()
            skipped += 1

    total_predicted = len(results)
    accuracy        = round((correct / total_predicted * 100), 2) if total_predicted > 0 else 0.0

    output = {
        'last_updated':    datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'total_matches':   total_predicted,
        'correct':         correct,
        'incorrect':       total_predicted - correct,
        'skipped':         skipped,
        'accuracy':        accuracy,
        'matches':         list(reversed(results))   # newest first
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Total predicted : {total_predicted}")
    print(f"  Correct         : {correct}")
    print(f"  Accuracy        : {accuracy}%")
    print(f"  Skipped         : {skipped}")
    print(f"  Saved to        : {OUTPUT_PATH}")
    print(f"{'='*50}")


if __name__ == '__main__':
    evaluate()
