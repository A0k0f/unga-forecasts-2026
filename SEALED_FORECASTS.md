# Sealed forecasts: December 2026 UN General Assembly plenary-only votes

These items go straight to the plenary, with no committee vote beforehand, so the outcome is not yet public. The forecasts come from each resolution's **title alone**, using a model trained on General Assembly votes up to December 2025.

```
created_utc: 2026-11-24T04:40Z
sha256(country.csv + summary.json + meta.json): 32d7a8183447529db2b87aa4f53edeba811a1544d471c9f8322519c96cd286d4
```

Tallies are attendance-adjusted: each country is weighted by its 2023–25 attendance rate, so they are comparable to official yes–no–abstain counts.

| Item | Forecast (yes–no–abstain) | Text-kNN baseline | Persistence baseline (copy last time) | Most uncertain countries | Poll first |
|---|---|---|---|---|---|
| Peaceful settlement of the question of Palestine | **158–6–11** | 157–7–11 | 155–10–11 | AU, CA, PG, KI, LR | CA, AU, AR |
| The Syrian Golan | **119–11–46** | 110–8–57 | 102–11–62 | GB, CZ, AT, HU, DE | GB, FR, AU |
| Jerusalem | **150–7–19** | 150–8–18 | 131–13–31 | AU, LR, GT, CA, CI | CA, LR, AU |
| Committee on the Exercise of the Inalienable Rights of the Palestinian People | **116–13–46** | 109–12–55 | 104–18–53 | GT, GB, LR, NZ, FR | GT, AR, CO |
| Division for Palestinian Rights of the Secretariat | **114–15–47** | 107–18–51 | 99–30–46 | GT, AR, LR, NZ, GB | GT, AR, LR |
| Special information programme on the question of Palestine (DGC) | **153–7–15** | 154–8–13 | 148–12–15 | LR, KI, GT, PG, HN | LR, GT, SB |
| Situation in the temporarily occupied territories of Ukraine | **96–15–64** | 100–15–61 | 88–18–70 | NE, AM, DZ, NG, IQ | ZW, IR, CN |

Notes:
- Expected non-voting is about 16 countries per vote. Venezuela is excluded because it has no recorded votes in 2023–25.
- The Ukraine item is the hardest test: its text is new (similarity 0.57 to the closest past resolution), so the forecast relies on the latent model rather than on precedent.
- Some items may not come to a vote this year (for example, if they are biennialized or merged). Only items that are actually voted on will be scored.
- Per-country probabilities are in sealed_forecasts_country.csv.