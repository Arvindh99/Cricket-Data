import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score,accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE_DIR, 'ipl_matches.csv'))

data.head()

team_name_mapping = {'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings',
                     'Royal Challengers Bangalore': 'Royal Challengers Bengaluru','Rising Pune Supergiants': 'Rising Pune Supergiant',
                     'Gujarat Lions':'Gujarat Titans'}

data['team1'] = data['team1'].replace(team_name_mapping)

defunct_teams = ['Deccan Chargers', 'Kochi Tuskers Kerala','Rising Pune Supergiant','Pune Warriors']

data = data[~data['team1'].isin(defunct_teams)]

data['team2'] = data['team2'].replace(team_name_mapping)

data = data[~data['team2'].isin(defunct_teams)]

data['toss_winner'] = data['toss_winner'].replace(team_name_mapping)
data['winner'] = data['winner'].replace(team_name_mapping)

data = data.drop(columns=['city'])

venue_mapping = {'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium','M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',
    'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium','MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
    'Feroz Shah Kotla': 'Arun Jaitley Stadium','Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
    'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium','Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
    'Wankhede Stadium, Mumbai': 'Wankhede Stadium','Eden Gardens, Kolkata': 'Eden Gardens',
    'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium','Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
    'Brabourne Stadium, Mumbai': 'Brabourne Stadium','Punjab Cricket Association Stadium, Mohali': 'IS Bindra Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'IS Bindra Stadium','Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'IS Bindra Stadium',
    'Punjab Cricket Association IS Bindra Stadium': 'IS Bindra Stadium','Sardar Patel Stadium, Motera': 'Narendra Modi Stadium',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 'Ekana Cricket Stadium',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam':'ACA-VDCA Cricket Stadium',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala':'HPCA Stadium',
    'Maharashtra Cricket Association Stadium, Pune':'MCA Stadium','Maharashtra Cricket Association Stadium':'MCA Stadium',
    "Sheikh Zayed Stadium": "Zayed Cricket Stadium, Abu Dhabi","Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
    "HPCA Stadium": "Himachal Pradesh Cricket Association Stadium","Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh":
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur","ACA-VDCA Cricket Stadium":"Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium"}

data['venue'] = data['venue'].replace(venue_mapping)

top_venues = data['venue'].value_counts().nlargest(20).index

data['venue'] = data['venue'].apply(lambda x: x if x in top_venues else 'Other')

data = data.drop(columns=['season'])

data.head()

"""Feature Engineering"""

data['toss_win_team1'] = (data['toss_winner'] == data['team1']).astype(int)

data['toss_decision'] = data['toss_decision'].map({'bat': 0,'field': 1})

data['team1_batting_first'] = (((data['toss_winner'] == data['team1']) & (data['toss_decision'] == 0)) |((data['toss_winner'] == data['team2']) & (data['toss_decision'] == 1))).astype(int)

team_win_rate = data.groupby('winner').size() / data.groupby('team1').size()
data['strength_diff'] = data['team1'].map(team_win_rate) - data['team2'].map(team_win_rate)

data['match_context'] = data['team1'].astype(str) + '_' + data['team2'].astype(str) + '_' + data['venue'].astype(str) + '_' + data['toss_winner'].astype(str)

context_wins = data.groupby(['match_context', 'winner']).size().unstack(fill_value=0)
context_matches = data.groupby('match_context').size()

def get_context_win_rate(row, team):
    ctx = row['match_context']
    t = row[team]
    if ctx in context_wins.index and t in context_wins.columns:
        return context_wins.loc[ctx, t] / context_matches[ctx]
    return 0.5

data['team1_context_win_prob'] = data.apply(lambda r: get_context_win_rate(r, 'team1'), axis=1)
data['team2_context_win_prob'] = data.apply(lambda r: get_context_win_rate(r, 'team2'), axis=1)

data = data.drop(columns=['match_context'])

encode = {'team1': {'Chennai Super Kings':1,'Royal Challengers Bengaluru':2,'Delhi Capitals':3,'Sunrisers Hyderabad':4,'Kolkata Knight Riders':5,'Lucknow Super Giants':6,'Mumbai Indians':7,'Punjab Kings':8,'Rajasthan Royals':9,'Gujarat Titans':10},
          'team2': {'Chennai Super Kings':1,'Royal Challengers Bengaluru':2,'Delhi Capitals':3,'Sunrisers Hyderabad':4,'Kolkata Knight Riders':5,'Lucknow Super Giants':6,'Mumbai Indians':7,'Punjab Kings':8,'Rajasthan Royals':9,'Gujarat Titans':10},
          'toss_winner': {'Chennai Super Kings':1,'Royal Challengers Bengaluru':2,'Delhi Capitals':3,'Sunrisers Hyderabad':4,'Kolkata Knight Riders':5,'Lucknow Super Giants':6,'Mumbai Indians':7,'Punjab Kings':8,'Rajasthan Royals':9,'Gujarat Titans':10},
          'winner': {'Chennai Super Kings':1,'Royal Challengers Bengaluru':2,'Delhi Capitals':3,'Sunrisers Hyderabad':4,'Kolkata Knight Riders':5,'Lucknow Super Giants':6,'Mumbai Indians':7,'Punjab Kings':8,'Rajasthan Royals':9,'Gujarat Titans':10}}

data.replace(encode, inplace=True)

cat_cols = ['venue']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

data = data.drop(columns=['win_by_runs','win_by_wickets'])

data = data.dropna()

data.head()

X = data.drop(columns=['winner'])
y = data['winner'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

algorithms = {
    'Logistic Regression': {
        "model": Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(solver='saga', max_iter=10000))]),
        "params": {
            "classifier__penalty": ['l1', 'l2', 'elasticnet'],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__l1_ratio": [0.0, 0.5, 1.0]
        }
    },
    'Decision Tree': {
        "model": Pipeline([('scaler', StandardScaler()), ('classifier', tree.DecisionTreeClassifier())]),
        "params": {
            "classifier__criterion": ['gini', 'entropy'],
            "classifier__max_depth": [None, 5, 10, 15, 20, 30],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8]
        }
    },
    'Random Forest': {
        "model": Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(random_state=42))]),
        "params": {
            "classifier__n_estimators": [100, 200, 300, 400],
            "classifier__max_features": ["sqrt", "log2", None],
            "classifier__max_depth": [None, 10, 20, 30, 40],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__bootstrap": [True, False]
        }
    },
    'NaiveBayes': {
        "model": Pipeline([('scaler', StandardScaler()), ('classifier', GaussianNB())]),
        "params": {
            "classifier__var_smoothing": np.logspace(0, -9, num=10)
        }
    },
    'K-Nearest Neighbors': {
        "model": Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())]),
        "params": {
            "classifier__n_neighbors": [3, 5, 7, 10, 15, 20],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        }
    },
    'Gradient Boost': {
        "model": Pipeline([('scaler', StandardScaler()), ('classifier', GradientBoostingClassifier(random_state=42))]),
        "params": {
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__n_estimators": [100, 200, 300],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_depth": [3, 5, 10],
            "classifier__subsample": [0.8, 0.9, 1.0]
        }
    }
}

prediction_models = {}
model_details = []

for model_name, values in algorithms.items():
    best_score = float('-inf')
    best_rscv = None

    try:
        rscv = RandomizedSearchCV(estimator=values["model"],param_distributions=values["params"],cv=5,n_iter=30,n_jobs=-1,verbose=0,random_state=42)
        rscv.fit(X_train, y_train)

        if rscv.best_score_ > best_score:
            best_score = rscv.best_score_
            best_rscv = rscv

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error with {model_name} (Prediction): {e}")
        break

    if best_rscv:
        prediction_models[model_name] = best_rscv
        model_details.append({"Model Name": model_name,"Best Score": best_score,"Best Parameters": best_rscv.best_params_})
        print(f"{model_name}: Best Score = {best_score:.4f}")
    else:
        print(f"{model_name}: No valid configuration found.")

pd.set_option('display.max_colwidth', None)
pd.DataFrame(model_details)

test_results = []

for model_name, model in prediction_models.items():
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    test_results.append({"Model Name": model_name,"Test Score": model.score(X_test, y_test),"Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],"F1-score": report["weighted avg"]["f1-score"]})

results_df = pd.DataFrame(test_results)

results_df

best_model_row = results_df.loc[results_df['Test Score'].idxmax()]
best_model_name = best_model_row['Model Name']
best_model = prediction_models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")
print(f"Test Score: {best_model_row['Test Score']:.4f}")


print("Saving model and preprocessors...")
joblib.dump(best_model, os.path.join(BASE_DIR, 'models', 'model', 'best_model.pkl'))
joblib.dump(encode, os.path.join(BASE_DIR, 'models', 'preprocess', 'encode.pkl'))
joblib.dump(label_encoders, os.path.join(BASE_DIR, 'models', 'preprocess', 'label_encoders.pkl'))
joblib.dump(team_name_mapping, os.path.join(BASE_DIR, 'models', 'mappings', 'team_name_mapping.pkl'))
joblib.dump(venue_mapping, os.path.join(BASE_DIR, 'models', 'mappings', 'venue_mapping.pkl'))
joblib.dump(top_venues, os.path.join(BASE_DIR, 'models', 'mappings', 'top_venues.pkl'))
joblib.dump(team_win_rate, os.path.join(BASE_DIR, 'models', 'stats', 'team_win_rate.pkl'))
joblib.dump(context_wins, os.path.join(BASE_DIR, 'models', 'stats', 'context_wins.pkl'))
joblib.dump(context_matches, os.path.join(BASE_DIR, 'models', 'stats', 'context_matches.pkl'))
joblib.dump(X.columns, os.path.join(BASE_DIR, 'models', 'preprocess', 'feature_columns.pkl'))
print("Export complete.")