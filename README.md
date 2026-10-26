# DSAI Club Induction 2026 — SY Task Round (Toxic Comment Classification)

## Name & Branch
- **Name:** Arnav Katepallewar
- **Branch:** CSE
- **Result:** public leaderboard mean ROC AUC **0.98775**; **990,006** trainable parameters (limit 1M)

## Model Architecture & Core Idea
- **Preprocessing:** raw text, no heavy cleaning; TF-IDF with sublinear term frequency, accents stripped, lowercased
- **Features:** word 1-3-grams (top 300k) + character 1-6-grams, `char_wb` (top 600k); vocabulary fit on training text only
- **Model:** one L2 logistic regression per label (6 independent binary classifiers, probabilities via `predict_proba`)
- **Naive-Bayes scaling (NB-LR):** each feature multiplied by log(p/q), the smoothed positive-vs-negative frequency ratio for that label
- **Multi-label setup:** toxic, severe_toxic, obscene, threat, insult, identity_hate each get their own model, features and regularisation strength C
- **Parameter count:** 6 labels x (165,000 weights + 1 bias) = 990,006 (checked from `coef_` / `intercept_` shapes; vectorizer vocabulary and NB ratios are fixed statistics, not trained weights)
- **Class imbalance:** metric is rank-based ROC AUC, so no resampling or class weights; stratified splits keep rare labels (threat ~0.3%) in every split
- **Validation:** stratified 80/20 hold-out (seed 42) for model choice and per-label C; stratified 5-fold CV for the earlier 140k-feature model (CV AUC 0.98726); every label-dependent step fitted on training folds only
- **Final training:** re-fit on 100% of `competition_train.csv`; no external data, no test queries to any model
- **Progression (mean AUC):** word TF-IDF baseline 0.9811 -> + char n-grams 0.9861 -> + NB scaling 0.9874 -> + per-label C (leaderboard 0.98722) -> + chi-squared selection (leaderboard 0.98775)

## Unique / Differentiating Ideas
- **Per-label chi-squared feature selection:** from ~900k candidate n-grams, each label keeps its own best 165,000, so the 1M budget goes to the most informative n-grams per label instead of the most frequent ones (gain +0.0005 on leaderboard)
- **Budget-driven design:** feature count chosen so the whole model sits just under 1M parameters (990,006)
- **Per-label regularisation:** `toxic` (most positives) uses C=1.0; rare labels (severe_toxic, threat, insult, identity_hate) use C=0.3
- **Character n-grams (1-6):** catch obfuscated or misspelled profanity that word vocabularies miss
- **Rejected ideas (tested, documented):** rank-blend of two models (+0.0002 but ~1.68M params, breaks limit); plain bigger vocabulary and leetspeak normalisation (`@ss` -> `ass`) gave no gain
- **Interpretable and fast:** linear model, trains in minutes on CPU, no pretrained embeddings or external data
