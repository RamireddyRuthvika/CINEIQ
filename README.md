# CINEIQ - Hybrid Movie Recommendation System

## Clone the Repository

```bash
git clone <repository-url>
cd CINEIQ
```

---

## Download Required Datasets

Download the following datasets:

- TMDB Movies Metadata (Kaggle) (https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?utm_source=chatgpt.com)
- MovieLens 25M Dataset (https://grouplens.org/datasets/movielens/25m)
- IMDb 50K Reviews Dataset (Kaggle) (https://grouplens.org/datasets/movielens/25m/?utm_source=chatgpt.com)

---

## Dataset Setup

Extract all datasets and organize them as follows:
Tis is directory tree:
```text
CINEIQ/
│
├── archive/
│   ├── credits.csv
│   ├── keywords.csv
│   ├── movies_metadata.csv
│   └── other TMDB metadata files
│
├── ml-25m/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── links.csv
│   ├── tags.csv
│   └── other MovieLens files
│
├── IMDB Dataset.csv
│
├── models/
│
├── requirements.txt
├── app.py
└── recommender.py
├──main.py
```

### Important

- Place all TMDB metadata files inside the `archive` folder.
- Place all MovieLens 25M files inside the `ml-25m` folder.
- Place `IMDB Dataset.csv` in the project root directory.
- Create an empty `models` folder before running the project.

---

## Create Virtual Environment

### Windows (Command Prompt)

```cmd
python -m venv venv
venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application
(terminal1)
```bash
streamlit run app.py
```
paralelly in another terminal run
(terminal2)
```bash
uvicorn main:app --reload
```
after application startup complete appears in  terminal2, paste the local host URL appeared in terminal1 in any browser(if browser is not auto opened) 
there browser app apperas after some time

---

## Features

- Content-Based Recommendation
- Collaborative Filtering
- Hybrid Recommendation System
- SVD-based Dimensionality Reduction
- Sentiment-Aware Ranking
- Explainable Recommendations
- Streamlit User Interface

  [demo_video](https://drive.google.com/file/d/12MxpTbiq8nD9-pwpJFO1wogGQ37dq4B1/view?usp=sharing)
