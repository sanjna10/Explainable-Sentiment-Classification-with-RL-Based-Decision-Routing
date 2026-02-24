# Explainable-Sentiment-Classification-with-RL-Based-Decision-Routing

# XAI + RL Triage for Sentiment Classification (IMDb)

A notebook project that combines **sentiment analysis**, **explainable AI (XAI)**, and a **cost-aware triage policy** for human-in-the-loop decision making.

## What this project does

- Trains a **TF-IDF + Logistic Regression** sentiment classifier on **IMDb labelled sentences**
- Explains predictions using:
  - **LIME** (local explanations)
  - **SHAP** (local feature attributions)
  - **ELI5 / model coefficients** (global feature importance, optional)
- Builds a **triage system** that decides whether to:
  - `auto_accept` (trust model)
  - `human_review`
  - `retrain_queue`
- Uses a **LinUCB contextual bandit** to learn better triage decisions under cost constraints

## Why it’s interesting

This project goes beyond basic classification by simulating a realistic ML deployment setup:
- confidence-aware automation
- explanation disagreement as a signal
- human review costs
- RL-inspired decision routing

## Tech Stack

- Python
- NumPy, Pandas, Matplotlib
- scikit-learn
- LIME 
- SHAP 
- ELI5 

## Dataset

Uses the **UCI IMDb labelled sentences** dataset (`imdb_labelled.txt`).

> Update the dataset path in the notebook if running locally.

## How to run

1. Open the notebook in Jupyter/Colab
2. Install dependencies
3. Place `imdb_labelled.txt` in the expected path
4. Run all cells top to bottom

## Install (core)

```bash
pip install numpy pandas matplotlib scikit-learn