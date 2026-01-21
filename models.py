import pandas as pd, numpy as np, requests, yaml, joblib
from xgboost import XGBClassifier

CFG = yaml.safe_load(open("config.yaml"))
HEAD = {"x-apisports-key": CFG["api_football_key"]}

def get_matches(league, season="2024"):
    url = f"https://v3.football.api-sports.io/fixtures?league={league}&season={season}"
    resp = requests.get(url, headers=HEAD).json()
    return pd.json_normalize(resp["response"])

def build_features(df):
    df["date"] = pd.to_datetime(df["fixture.date"])
    df["hGoals"] = pd.to_numeric(df["goals.home"], errors="coerce").fillna(0)
    df["aGoals"] = pd.to_numeric(df["goals.away"], errors="coerce").fillna(0)
    df["hLastGoals"]   = df.groupby("teams.home.id")["hGoals"].shift(1).rolling(5, min_periods=1).mean()
    df["aLastGoals"]   = df.groupby("teams.away.id")["aGoals"].shift(1).rolling(5, min_periods=1).mean()
    # escanteios
    df["hLastCorners"] = df.groupby("teams.home.id")["statistics"].apply(
        lambda x: (x[0]["statistics"][6]["value"] or 0) if len(x)>0 else 0).rolling(5, min_periods=1).mean()
    df["aLastCorners"] = df.groupby("teams.away.id")["statistics"].apply(
        lambda x: (x[1]["statistics"][6]["value"] or 0) if len(x)>1 else 0).rolling(5, min_periods=1).mean()
    # cartões
    df["hLastCards"]   = df.groupby("teams.home.id")["statistics"].apply(
        lambda x: (x[0]["statistics"][4]["value"] or 0) if len(x)>0 else 0).rolling(5, min_periods=1).mean()
    df["aLastCards"]   = df.groupby("teams.away.id")["statistics"].apply(
        lambda x: (x[1]["statistics"][4]["value"] or 0) if len(x)>1 else 0).rolling(5, min_periods=1).mean()
      # alvos
    df["target_over15"]      = (df["hGoals"] + df["aGoals"] > 1.5).astype(int)
    df["target_btts"]        = ((df["hGoals"] > 0) & (df["aGoals"] > 0)).astype(int)
    df["target_corners_over95"] = (df["hLastCorners"] + df["aLastCorners"] > 9.5).astype(int)
    df["target_cards_over25"]   = (df["hLastCards"]   + df["aLastCards"]   > 2.5).astype(int)
    return df.dropna(subset=["target_over15"])

def train():
    final = pd.concat([build_features(get_matches(lg)) for lg in CFG["leagues"]])
    feats = ["hLastGoals","aLastGoals","hLastCorners","aLastCorners","hLastCards","aLastCards"]
    X = final[feats].fillna(0)
    for t in ["target_over15","target_btts","target_corners_over95","target_cards_over25"]:
        mod = XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,objective="binary:logistic")
        mod.fit(X, final[t])
        joblib.dump(mod, f"{t}.pkl")
    print("Modelos 70 % treinados ✅")

if __name__ == "__main__":
    train()
