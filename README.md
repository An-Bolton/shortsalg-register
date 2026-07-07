# Shortsalgregister

Dette her er et Streamlit-basert shortregister for Oslo Børs basert på åpne data fra Finanstilsynet.

## Funksjoner her er som følger:

- Oversikt over shortposisjoner
- Søke og filtrere
- Se på historiske data
- Ulike og diverse isualiseringer av short-utvikling

## Kjør lokalt

```bash
pip install -r requirements.txt
streamlit run shortsalg_app.py
```

## Kjør med Docker

```bash
docker build -t shortregister .
docker run -p 8501:8501 shortregister
```

## Datakilde

Åpne data fra Finanstilsynet.