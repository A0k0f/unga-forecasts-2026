"""Live forecasting: fit TCIP on ALL available votes (1991 - Dec 2025) and forecast new resolutions.

  python src/live.py fit                       # train final model -> results_live/tcip_live.pkl
  python src/live.py forecast items.csv out    # items.csv: id,title,date[,issue] -> forecasts
  python src/live.py score forecasts actual.csv # actual.csv: id,ms_name,vote (Y/N/A/X)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B
import bellwether as BW
import data as D
import metrics
import tcip
from features import RollCallFeaturizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_live")
os.makedirs(OUT, exist_ok=True)
# Hyper-parameters selected on the 2017-19 validation window (results_window2020/tuning.json)
HP = {"d": 12, "mixture": {"K": 20, "p": 4, "w_text": 0.2, "s1": 0.3}, "knn_K": 5}


def fit():
    t0 = time.time()
    D.EXTRA_CSV = os.path.join(ROOT, "data", "voting_all_countries_2010_2025.csv")
    D.TRAIN_SESSIONS, D.VAL_SESSIONS, D.TEST_SESSIONS = (46, 80), (98, 98), (99, 99)
    d = D.load()
    rc = d.rolls[d.rolls.split == "train"].reset_index(drop=True)
    cidx = {c: i for i, c in enumerate(d.countries)}
    ridx = {r: i for i, r in enumerate(rc.rcid)}
    v = d.votes[d.votes.rcid.isin(ridx)].copy()
    v["ri"], v["ci"] = v.rcid.map(ridx), v.country_code.map(cidx)
    M = B.vote_tensor(v, ridx, len(cidx))
    fz = RollCallFeaturizer().fit(rc)
    X = fz.transform(rc)
    m = tcip.TCIP(d=HP["d"], maxiter=1500, verbose=True).fit(v.ci.values, v.ri.values, v.y.values, X, len(cidx))
    # current members = voted in the last 3 sessions
    recent = rc.session.values >= rc.session.max() - 2
    active = (M[recent] >= 0).any(0)
    pickle.dump({"model": m, "featurizer": fz, "rolls": rc[["rcid", "session", "year", "short", "text", "unres"]],
                 "M": M, "countries": d.countries, "names": d.country_names, "active": active, "hp": HP},
                open(os.path.join(OUT, "tcip_live.pkl"), "wb"))
    print(f"fit on {len(rc)} roll calls / {len(v)} votes, sessions {rc.session.min()}-{rc.session.max()}, "
          f"{active.sum()} active members, {time.time() - t0:.0f}s")


class Live:
    def __init__(self):
        b = pickle.load(open(os.path.join(OUT, "tcip_live.pkl"), "rb"))
        self.__dict__.update(b)
        self.m = b["model"]
        self.T = self.featurizer.vec.transform(self.rolls["text"].values)
        self.mp = tcip.MixturePosterior(self.m, **HP["mixture"])
        self.base = B.country_prior(1, self.M)[0]
        self.onehot = np.stack([(self.M == j) for j in range(3)], -1).astype(float)

    def featurize(self, title, issue=None):
        row = {"text": D.clean_text(title), "importantvote": 0, "amend": 0, "para": 0}
        for k in ["co", "di", "ec", "hr", "me", "nu"]:
            row["issue_" + k] = int(issue == k)
        row["issue_none"] = int(issue is None)
        df = pd.DataFrame([row])
        return self.featurizer.transform(df), self.featurizer.tfidf(df)

    def predict(self, title, issue=None, observed=None):
        """Returns dict of model predictive (C x 3), kNN and persistence baselines, analogues, BALD ranking."""
        X, T = self.featurize(title, issue)
        sim = (T @ self.T.T).toarray()[0]
        mu = self.m.prior_mean(X)[0]
        observed = observed or {}
        ci = {c: i for i, c in enumerate(self.countries)}
        oc = np.array([ci[c] for c in observed], int)
        oy = np.array([D.VOTE_CODE[v] for v in observed.values()], int)
        zs, Hs, w = self.mp.update(self.mp.components(sim, mu), oc, oy)
        P = self.mp.predictive(zs, w, Hs)
        nn = np.argsort(-sim)[: HP["knn_K"]]
        wk = np.clip(sim[nn], 0, None) ** 2 + 1e-6
        Pk = (wk[:, None, None] * self.onehot[nn]).sum(0) + 0.5 * self.base
        Pk /= Pk.sum(-1, keepdims=True)
        j = nn[0]                                     # persistence: copy votes on the single most similar past roll call
        Pp = np.where(self.M[j][:, None] >= 0, 0.02, 0.0) + self.onehot[j] * 0.94
        Pp = np.where(self.M[j][:, None] >= 0, Pp, self.base)
        Pp /= Pp.sum(-1, keepdims=True)
        cand = np.where(self.active)[0]
        bald = BW.bald_scores(self.mp, zs, Hs, w, cand)
        order = cand[np.argsort(-bald)]
        analog = [(int(self.rolls.year.iloc[k]), str(self.rolls.short.iloc[k])[:90], float(sim[k])) for k in nn[:3]]
        return {"P": P, "P_knn": Pk, "P_persist": Pp, "analogues": analog, "bald_order": [self.countries[c] for c in order[:10]],
                "max_sim": float(sim.max())}


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit()
