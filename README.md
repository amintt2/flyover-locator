# Flyover Locator (dopplergeo)

Localise un enregistrement audio/vidéo à partir des courbes Doppler des avions
à hélice qu'on entend passer, comparées aux trajectoires ADS-B réelles.

## Comment ça marche

1. **Spectrogramme boosté** : chaque fréquence est normalisée par rapport à son
   bruit de fond, donc une courbe Doppler inaudible ressort.
2. **Détection auto des passages** : on suit la ligne la plus forte et on fitte
   le modèle Doppler exact (temps retardé). On en tire la vitesse, la distance
   au plus proche passage (CPA) et l'instant du CPA.
3. **ADS-B** : l'archive journalière adsb.lol (~3-4 Go) est lue une seule fois en
   streaming. On ne garde que ce qui peut s'entendre (sous 6000 m, pas de jets
   connus, un point toutes les 4 s) : ~280 Mo par jour, dans `/data/adsb_index`.
   Chaque recherche suivante sur ce jour lit cet index au lieu de l'archive.
4. **Recherche grossière** (grille de 200 m) : on garde les points qui ont
   entendu un avion à hélice de la bonne vitesse, à la bonne distance, puis le
   suivant avec le bon écart de temps, sans aucun autre avion proche entre les deux.
5. **Score fin** (grille de 100 m) : pour chaque point, on modélise la courbe
   Doppler exacte produite par la vraie trajectoire de l'avion, puis on la
   compare au spectrogramme avec l'écart de temps verrouillé.
6. **Altitude du sol** : tuiles SRTM téléchargées automatiquement (AWS skadi).
7. **Trouver le jour** : sans date, on donne une période. Chaque jour est
   cherché en parallèle et tous les passages nets doivent coller. Le bon jour
   ressort nettement devant les autres.

## Préparer l'index à l'avance

Un jour pas encore indexé prend ~6-7 min la première fois. Pour préparer une
période en fond (2 jours à la fois, limité par le réseau) :

```bash
python -m dopplergeo.index 2026-10-01 2026-10-31 --workers 2
```

## Lancer en local

```bash
uv venv && uv pip install -e ".[web,dev]"
uvicorn dopplergeo.web:app --port 8000
```

Test de bout en bout sur un scénario synthétique (≈ 40 s) :

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -s
```

## Déploiement (Coolify)

Build pack « Dockerfile », port 8000, volume persistant sur `/data`.

Variables d'environnement :
- `APP_PASSWORD` : mot de passe partagé, fortement conseillé si le site est public.
- `MAX_RADIUS_KM` : rayon de recherche max (100 par défaut).
- `MAX_UPLOAD_MB` : taille max d'upload (500 par défaut).
- `MAX_DAYS` : période max pour « Trouver le jour » (31 par défaut).
- `DAY_WORKERS` : jours cherchés en même temps (cœurs - 1, max 4, par défaut).

## Limites

- Sans date, il faut au moins **2 passages d'avion nets** pour retrouver le jour.
- Seuls les **avions à hélice** sont exploités. Les jets sont trop hauts et leur bruit est large bande.
- Couverture ADS-B : les données adsb.lol sont collectées par des bénévoles et peuvent être trouées dans les zones rurales.
- À utiliser sur tes propres enregistrements ou dans un cadre d'enquête légitime.
