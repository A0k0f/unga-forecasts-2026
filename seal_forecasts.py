"""Write sealed (time-stamped, hashed) forecasts for December 2026 plenary-only General Assembly votes."""
import hashlib, json, os, sys, datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live import Live, OUT
import data as D

ITEMS = [  # plenary-only items (no Main Committee vote); titles as used in previous sessions
    ("PAL-1", "Peaceful settlement of the question of Palestine", "me"),
    ("GOLAN", "The Syrian Golan", "me"),
    ("JERUS", "Jerusalem", "me"),
    ("PAL-2", "Committee on the Exercise of the Inalienable Rights of the Palestinian People", "me"),
    ("PAL-3", "Division for Palestinian Rights of the Secretariat", "me"),
    ("PAL-4", "Special information programme on the question of Palestine of the Department of Global Communications of the Secretariat", "me"),
    ("UKR", "Situation in the temporarily occupied territories of Ukraine", None),
]
L = Live()
codes = [c for c, a in zip(L.countries, L.active) if a]
act = np.where(L.active)[0]
recent = L.rolls.session.values >= 78                        # sessions 78-80 (2023-2025)
rho = (L.M[recent][:, act] >= 0).mean(0)                     # attendance rate per current member
rows, summ = [], []
for iid, title, issue in ITEMS:
    r = L.predict(title, issue)
    for name, P in [("TCIP-Mix", r["P"]), ("Text kNN", r["P_knn"]), ("Persistence", r["P_persist"])]:
        Pa = P[act]
        if name == "TCIP-Mix":
            for c, p in zip(codes, Pa):
                rows.append({"item": iid, "country": c, "name": L.names.get(c, c), "p_no": round(p[0], 4),
                             "p_abstain": round(p[1], 4), "p_yes": round(p[2], 4), "forecast": D.VOTE_NAMES[p.argmax()]})
        t = np.bincount(Pa.argmax(1), minlength=3)
        summ.append({"item": iid, "title": title, "model": name, "yes": int(t[2]), "no": int(t[0]), "abstain": int(t[1]),
                     "exp_yes": round(float(Pa[:, 2].sum()), 1), "exp_no": round(float(Pa[:, 0].sum()), 1),
                     "exp_abstain": round(float(Pa[:, 1].sum()), 1),
                     "att_adj_yes": round(float((rho * Pa[:, 2]).sum())), "att_adj_no": round(float((rho * Pa[:, 0]).sum())),
                     "att_adj_abstain": round(float((rho * Pa[:, 1]).sum())), "att_adj_not_voting": round(float((1 - rho).sum()))})
    Pa = r["P"][act]
    ent = -(Pa * np.log(Pa)).sum(1)
    swing = [codes[i] for i in np.argsort(-ent)[:10]]
    summ[-3].update({"max_text_similarity": round(r["max_sim"], 3), "analogues": r["analogues"],
                     "most_uncertain_countries": swing, "poll_first_BALD": r["bald_order"][:5]})
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
pc = pd.DataFrame(rows); sm = pd.DataFrame(summ)
pc.to_csv(os.path.join(OUT, "sealed_forecasts_country.csv"), index=False)
meta = {"created_utc": stamp, "model": "TCIP-Mix trained on UNGA votes 1991-Dec 2025 (sessions 46-80)",
        "hyperparameters": L.hp, "items": [dict(zip(["id", "title", "issue"], it)) for it in ITEMS],
        "note": "Forecasts from resolution titles only; no committee votes or 2026 votes used. Argmax/expected tallies are over 192 current members (Venezuela excluded: no votes recorded in 2023-25). att_adj_* tallies weight each country by its 2023-25 attendance rate, to compare with official yes-no-abstain counts."}
sm.to_json(os.path.join(OUT, "sealed_forecasts_summary.json"), orient="records", indent=1)
json.dump(meta, open(os.path.join(OUT, "sealed_forecasts_meta.json"), "w"), indent=1)
h = hashlib.sha256()
for f in ["sealed_forecasts_country.csv", "sealed_forecasts_summary.json", "sealed_forecasts_meta.json"]:
    h.update(open(os.path.join(OUT, f), "rb").read())
open(os.path.join(OUT, "SEAL.txt"), "w").write(f"created_utc: {stamp}\nsha256(country.csv + summary.json + meta.json): {h.hexdigest()}\n")
print(open(os.path.join(OUT, "SEAL.txt")).read())
pd.set_option("display.width", 220)
print(sm[["item", "model", "yes", "no", "abstain", "att_adj_yes", "att_adj_no", "att_adj_abstain", "att_adj_not_voting"]].to_string(index=False))
for s in summ[::3]:
    print(s["item"], "| sim", s["max_text_similarity"], "| analog:", s["analogues"][0][:2], "| uncertain:", s["most_uncertain_countries"][:6], "| poll:", s["poll_first_BALD"])
