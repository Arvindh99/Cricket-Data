import pandas as pd
import numpy as np
import os
import traceback
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for folder in ['models/model', 'models/preprocess', 'models/mappings', 'models/stats']:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)


data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'csv', 'ipl_matches.csv'))

team_name_mapping = {
    'Delhi Daredevils':            'Delhi Capitals',
    'Kings XI Punjab':             'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiants':     'Rising Pune Supergiant',
    'Gujarat Lions':               'Gujarat Titans'
}

defunct_teams = ['Deccan Chargers', 'Kochi Tuskers Kerala', 'Rising Pune Supergiant', 'Pune Warriors']

for col in ['team1', 'team2', 'toss_winner', 'winner']:
    data[col] = data[col].replace(team_name_mapping)

data = data[~data['team1'].isin(defunct_teams)]
data = data[~data['team2'].isin(defunct_teams)]
data = data[~data['winner'].isin(defunct_teams)]
data = data.reset_index(drop=True)

data = data.drop(columns=['city', 'season'], errors='ignore')
swap_mask = data['team1'] > data['team2']
data.loc[swap_mask, ['team1', 'team2']] = (data.loc[swap_mask, ['team2', 'team1']].values)

venue_mapping = {
    'M.Chinnaswamy Stadium':                                                     'M Chinnaswamy Stadium',
    'M Chinnaswamy Stadium, Bengaluru':                                          'M Chinnaswamy Stadium',
    'MA Chidambaram Stadium, Chepauk':                                           'MA Chidambaram Stadium',
    'MA Chidambaram Stadium, Chepauk, Chennai':                                  'MA Chidambaram Stadium',
    'Feroz Shah Kotla':                                                          'Arun Jaitley Stadium',
    'Arun Jaitley Stadium, Delhi':                                               'Arun Jaitley Stadium',
    'Rajiv Gandhi International Stadium, Uppal':                                 'Rajiv Gandhi International Stadium',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad':                     'Rajiv Gandhi International Stadium',
    'Wankhede Stadium, Mumbai':                                                  'Wankhede Stadium',
    'Eden Gardens, Kolkata':                                                     'Eden Gardens',
    'Sawai Mansingh Stadium, Jaipur':                                            'Sawai Mansingh Stadium',
    'Dr DY Patil Sports Academy, Mumbai':                                        'Dr DY Patil Sports Academy',
    'Brabourne Stadium, Mumbai':                                                 'Brabourne Stadium',
    'Punjab Cricket Association Stadium, Mohali':                                'IS Bindra Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali':                     'IS Bindra Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh':         'IS Bindra Stadium',
    'Punjab Cricket Association IS Bindra Stadium':                              'IS Bindra Stadium',
    'Sardar Patel Stadium, Motera':                                              'Narendra Modi Stadium',
    'Narendra Modi Stadium, Ahmedabad':                                          'Narendra Modi Stadium',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow':   'Ekana Cricket Stadium',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam':      'ACA-VDCA Cricket Stadium',
    'ACA-VDCA Cricket Stadium':                                                  'ACA-VDCA Cricket Stadium',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala':                 'HPCA Stadium',
    'HPCA Stadium':                                                              'HPCA Stadium',
    'Maharashtra Cricket Association Stadium, Pune':                            'MCA Stadium',
    'Maharashtra Cricket Association Stadium':                                   'MCA Stadium',
    'Sheikh Zayed Stadium':                                                      'Zayed Cricket Stadium, Abu Dhabi',
    'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh':  'Maharaja Yadavindra Singh International Cricket Stadium',
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur':  'Maharaja Yadavindra Singh International Cricket Stadium',
}

data['venue'] = data['venue'].replace(venue_mapping)
top_venues    = data['venue'].value_counts().nlargest(20).index
data['venue'] = data['venue'].apply(lambda x: x if x in top_venues else 'Other')

data['toss_decision']       = data['toss_decision'].map({'bat': 0, 'field': 1})
data['toss_win_team1']      = (data['toss_winner'] == data['team1']).astype(int)
data['team1_batting_first'] = (
    ((data['toss_winner'] == data['team1']) & (data['toss_decision'] == 0)) |
    ((data['toss_winner'] == data['team2']) & (data['toss_decision'] == 1))
).astype(int)

HIGH_TOSS_IMPACT_VENUES = {
    # UAE — most decisive toss effect in IPL history
    'Dubai International Cricket Stadium',
    'Sharjah Cricket Stadium',
    'Zayed Cricket Stadium, Abu Dhabi',

    # Coastal / high-dew Indian venues
    'Wankhede Stadium',                    # Mumbai — strongest dew in IPL
    'MA Chidambaram Stadium',              # Chennai — dew + turning pitch
    'Eden Gardens',                        # Kolkata — dew + slow surface
    'Rajiv Gandhi International Stadium',  # Hyderabad — consistently heavy dew
    'M Chinnaswamy Stadium',               # Bangalore — elevation + dew combo
    'Brabourne Stadium',                   # Mumbai — same coastal conditions
    'Dr DY Patil Sports Academy',          # Navi Mumbai — coastal dew

    # Significant dew in recent IPL seasons
    'Narendra Modi Stadium',               # Ahmedabad — day-night dew factor
    'Ekana Cricket Stadium',               # Lucknow — heavy dew reported
}

# Toss advantage only meaningful at high-dew venues — interaction term
data['toss_matters']             = data['venue'].isin(HIGH_TOSS_IMPACT_VENUES).astype(int)
data['effective_toss_advantage'] = data['toss_win_team1'] * data['toss_matters']


train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=42)
train_mask = data.index.isin(train_idx)

print(f"Train samples : {train_mask.sum()} | Test samples : {(~train_mask).sum()}")

train_data = data[train_mask].copy()

total_wins    = train_data.groupby('winner').size()
total_matches = (train_data.groupby('team1').size()
                 .add(train_data.groupby('team2').size(), fill_value=0))
team_win_rate = (total_wins / total_matches).fillna(0)

data['strength_diff'] = (
    data['team1'].map(team_win_rate).fillna(0) -
    data['team2'].map(team_win_rate).fillna(0)
)

h2h_wins  = train_data.groupby(['team1', 'team2', 'winner']).size().unstack(fill_value=0)
h2h_total = train_data.groupby(['team1', 'team2']).size()

def get_h2h_win_rate(row):
    key = (row['team1'], row['team2'])
    if key in h2h_total.index and row['team1'] in h2h_wins.columns:
        wins  = h2h_wins.loc[key, row['team1']] if key in h2h_wins.index else 0
        total = h2h_total[key]
        return wins / total if total > 0 else 0.5
    return 0.5

data['h2h_win_rate_team1'] = data.apply(get_h2h_win_rate, axis=1)

venue_wins  = train_data.groupby(['venue', 'winner']).size().unstack(fill_value=0)
venue_total = train_data.groupby('venue').size()

def get_venue_win_rate(team, venue):
    if venue in venue_total.index and team in venue_wins.columns:
        wins  = venue_wins.loc[venue, team] if venue in venue_wins.index else 0
        total = venue_total[venue]
        return wins / total if total > 0 else 0.5
    return 0.5

data['team1_venue_win_rate'] = data.apply(lambda r: get_venue_win_rate(r['team1'], r['venue']), axis=1)
data['team2_venue_win_rate'] = data.apply(lambda r: get_venue_win_rate(r['team2'], r['venue']), axis=1)

data['match_context'] = (
    data['team1'].astype(str) + '_' +
    data['team2'].astype(str) + '_' +
    data['venue'].astype(str) + '_' +
    data['toss_winner'].astype(str)
)

_train_ctx  = train_data.assign(
    match_context=(
        train_data['team1'].astype(str) + '_' +
        train_data['team2'].astype(str) + '_' +
        train_data['venue'].astype(str) + '_' +
        train_data['toss_winner'].astype(str)
    )
)
ctx_wins    = _train_ctx.groupby(['match_context', 'winner']).size().unstack(fill_value=0)
ctx_matches = _train_ctx.groupby('match_context').size()

def get_context_win_rate(row, team_col):
    ctx  = row['match_context']
    team = row[team_col]
    if ctx in ctx_wins.index and team in ctx_wins.columns:
        return ctx_wins.loc[ctx, team] / ctx_matches[ctx]
    return 0.5

data['team1_context_win_prob'] = data.apply(lambda r: get_context_win_rate(r, 'team1'), axis=1)

data = data.drop(columns=['match_context'])

current_season_path = os.path.join(BASE_DIR, 'data', 'csv', 'ipl_current_season.csv')

if os.path.exists(current_season_path):
    print("\nLoading current season data for team form...")
    curr = pd.read_csv(current_season_path)

    for col in ['team1', 'team2', 'winner']:
        if col in curr.columns:
            curr[col] = curr[col].replace(team_name_mapping)

    # Stack into one row per (date, team, won) so we can compute cumulative form
    all_curr = pd.concat([
        curr[['team1', 'winner']].rename(columns={'team1': 'team'}),
        curr[['team2', 'winner']].rename(columns={'team2': 'team'})
    ]).reset_index(drop=True)

    all_curr['won'] = (all_curr['team'] == all_curr['winner']).astype(int)

    # Expanding (cumulative) mean shifted by 1 so a match doesn't count itself
    all_curr['curr_season_form'] = (
        all_curr.groupby('team')['won']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        .fillna(0.5)
    )

    # Latest form value per team — used both here and at inference time
    latest_form = all_curr.groupby('team')['curr_season_form'].last().fillna(0.5)

    # Map onto historical dataset (training rows get season-level signal)
    data['team1_form'] = data['team1'].map(latest_form).fillna(
        data['team1'].map(team_win_rate).fillna(0.5)
    )
    data['team2_form'] = data['team2'].map(latest_form).fillna(
        data['team2'].map(team_win_rate).fillna(0.5)
    )

    print(f"  Current season form loaded for {len(latest_form)} teams:")
    for team, form in latest_form.sort_values(ascending=False).items():
        print(f"    {team:<45} {form:.3f}")

else:
    print("\nWarning: ipl_current_season.csv not found — falling back to historical win rate for form.")
    data['team1_form'] = data['team1'].map(team_win_rate).fillna(0.5)
    data['team2_form'] = data['team2'].map(team_win_rate).fillna(0.5)
    latest_form = team_win_rate.copy()

joblib.dump(latest_form, os.path.join(BASE_DIR, 'models', 'stats', 'current_season_form.pkl'))
print("  current_season_form.pkl saved.")


team_categories = sorted(set(data['team1'].tolist() + data['team2'].tolist()))
encode = {team: idx for idx, team in enumerate(team_categories)} 

for col in ['team1', 'team2', 'toss_winner']:
    data[col] = data[col].map(encode).fillna(-1).astype(int)

data['winner'] = data['winner'].map(encode).fillna(-1).astype(int)
data = data[data['winner'] >= 0]

label_encoders = {}
for col in ['venue']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

data = data.drop(columns=['win_by_runs', 'win_by_wickets'], errors='ignore')
if 'date' in data.columns:
    data = data.drop(columns=['date'])

data = data.dropna().reset_index(drop=True)

print(f"\nFinal dataset shape : {data.shape}")
print(f"Class distribution  :\n{data['winner'].value_counts().sort_index()}")

X = data.drop(columns=['winner'])
y = data['winner'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTotal features ({len(X.columns)}): {list(X.columns)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

algorithms = {
    'Logistic Regression': {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(solver='saga', max_iter=10000))
        ]),
        "params": {
            "classifier__penalty":  ['l1', 'l2', 'elasticnet'],
            "classifier__C":        [0.01, 0.1, 1, 10, 100],
            "classifier__l1_ratio": [0.0, 0.5, 1.0]
        }
    },
    'Decision Tree': {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', tree.DecisionTreeClassifier())
        ]),
        "params": {
            "classifier__criterion":         ['gini', 'entropy'],
            "classifier__max_depth":         [None, 5, 10, 15, 20, 30],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf":  [1, 2, 4, 8]
        }
    },
    'Random Forest': {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        "params": {
            "classifier__n_estimators":      [100, 200, 300, 400],
            "classifier__max_features":      ["sqrt", "log2", None],
            "classifier__max_depth":         [None, 10, 20, 30, 40],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__bootstrap":         [True, False]
        }
    },
    'NaiveBayes': {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ]),
        "params": {
            "classifier__var_smoothing": np.logspace(0, -9, num=10)
        }
    },
    'K-Nearest Neighbors': {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ]),
        "params": {
            "classifier__n_neighbors": [3, 5, 7, 10, 15, 20],
            "classifier__weights":     ["uniform", "distance"],
            "classifier__p":           [1, 2]
        }
    },
    'Gradient Boost': {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ]),
        "params": {
            "classifier__learning_rate":     [0.01, 0.05, 0.1, 0.2],
            "classifier__n_estimators":      [100, 200, 300],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_depth":         [3, 5, 10],
            "classifier__subsample":         [0.8, 0.9, 1.0]
        }
    },
    'XGBoost': {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=42
            ))
        ]),
        "params": {
            "classifier__n_estimators":     [100, 200, 300],
            "classifier__max_depth":        [3, 5, 7],
            "classifier__learning_rate":    [0.01, 0.05, 0.1],
            "classifier__subsample":        [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.9, 1.0],
            "classifier__reg_alpha":        [0, 0.1, 0.5],
            "classifier__reg_lambda":       [1, 1.5, 2]
        }
    },
    'LightGBM': {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LGBMClassifier(random_state=42, verbose=-1))
        ]),
        "params": {
            "classifier__n_estimators":     [100, 200, 300],
            "classifier__max_depth":        [3, 5, 7, -1],
            "classifier__learning_rate":    [0.01, 0.05, 0.1],
            "classifier__num_leaves":       [20, 31, 50],
            "classifier__subsample":        [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.9, 1.0]
        }
    }
}


prediction_models = {}
model_details     = []

for model_name, values in algorithms.items():
    print(f"\nTraining: {model_name} ...")
    try:
        rscv = RandomizedSearchCV(
            estimator=values["model"],
            param_distributions=values["params"],
            cv=cv,
            n_iter=30,
            n_jobs=-1,
            verbose=0,
            random_state=42,
            scoring='accuracy'
        )
        rscv.fit(X_train, y_train)

        prediction_models[model_name] = rscv
        model_details.append({
            "Model Name":      model_name,
            "CV Best Score":   rscv.best_score_,
            "Best Parameters": rscv.best_params_
        })
        print(f"{model_name}: CV Score = {rscv.best_score_:.4f}")

    except Exception as e:
        print(f"Error with {model_name}: {e}")
        traceback.print_exc()
        continue


print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)

test_results = []

for model_name, model in prediction_models.items():
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    report  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        auc = None

    test_results.append({
        "Model Name": model_name,
        "Test Acc":   model.score(X_test, y_test),
        "ROC-AUC":    auc,
        "Precision":  report["weighted avg"]["precision"],
        "Recall":     report["weighted avg"]["recall"],
        "F1-score":   report["weighted avg"]["f1-score"]
    })

results_df = pd.DataFrame(test_results).sort_values("Test Acc", ascending=False)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.4f}'.format)
print(results_df.to_string(index=False))

best_model_row  = results_df.iloc[0]
best_model_name = best_model_row['Model Name']
best_model      = prediction_models[best_model_name]

print(f"\nBest Model Selected : {best_model_name}")
print(f"   Test Accuracy       : {best_model_row['Test Acc']:.4f}")
if best_model_row['ROC-AUC']:
    print(f"   ROC-AUC (weighted)  : {best_model_row['ROC-AUC']:.4f}")

team_names_ordered = [t for t, _ in sorted(encode.items(), key=lambda x: x[1])]

fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test,
    display_labels=team_names_ordered,
    xticks_rotation=45,
    ax=ax
)
ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'models', 'confusion_matrix.png'), dpi=150)
plt.close()
print("\nConfusion matrix saved to models/confusion_matrix.png")

try:
    clf = best_model.best_estimator_.named_steps['classifier']
    if hasattr(clf, 'feature_importances_'):
        importances = pd.Series(clf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        importances.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'Feature Importances — {best_model_name}', fontsize=13)
        ax.set_ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'models', 'feature_importances.png'), dpi=150)
        plt.close()
        print("Feature importance plot saved to models/feature_importances.png")
        print(f"\nTop 10 Features:\n{importances.head(10)}")
except Exception as e:
    print(f"Feature importance not available: {e}")

print("\nSaving model and preprocessors...")

joblib.dump(best_model,              os.path.join(BASE_DIR, 'models', 'model',      'best_model.pkl'))
joblib.dump(encode,                  os.path.join(BASE_DIR, 'models', 'preprocess', 'encode.pkl'))
joblib.dump(label_encoders,          os.path.join(BASE_DIR, 'models', 'preprocess', 'label_encoders.pkl'))
joblib.dump(team_categories,         os.path.join(BASE_DIR, 'models', 'preprocess', 'team_categories.pkl'))
joblib.dump(X.columns,               os.path.join(BASE_DIR, 'models', 'preprocess', 'feature_columns.pkl'))

joblib.dump(team_name_mapping,       os.path.join(BASE_DIR, 'models', 'mappings',   'team_name_mapping.pkl'))
joblib.dump(venue_mapping,           os.path.join(BASE_DIR, 'models', 'mappings',   'venue_mapping.pkl'))
joblib.dump(top_venues,              os.path.join(BASE_DIR, 'models', 'mappings',   'top_venues.pkl'))
joblib.dump(HIGH_TOSS_IMPACT_VENUES, os.path.join(BASE_DIR, 'models', 'mappings',   'high_toss_impact_venues.pkl'))

joblib.dump(team_win_rate,           os.path.join(BASE_DIR, 'models', 'stats',      'team_win_rate.pkl'))
joblib.dump(ctx_wins,                os.path.join(BASE_DIR, 'models', 'stats',      'context_wins.pkl'))
joblib.dump(ctx_matches,             os.path.join(BASE_DIR, 'models', 'stats',      'context_matches.pkl'))
joblib.dump(h2h_wins,                os.path.join(BASE_DIR, 'models', 'stats',      'h2h_wins.pkl'))
joblib.dump(h2h_total,               os.path.join(BASE_DIR, 'models', 'stats',      'h2h_total.pkl'))
joblib.dump(venue_wins,              os.path.join(BASE_DIR, 'models', 'stats',      'venue_wins.pkl'))
joblib.dump(venue_total,             os.path.join(BASE_DIR, 'models', 'stats',      'venue_total.pkl'))

print("Export complete.")