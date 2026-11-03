# Flyover Locator (dopplergeo)

Localise un enregistrement audio/vidéo à partir des courbes Doppler des avions
à hélice qu'on entend passer, comparées aux trajectoires ADS-B réelles.

## Comment ça marche

1. **Spectrogramme boosté** : chaque fréquence est normalisée par rapport à son
   bruit de fond, donc une courbe Doppler inaudible ressort.
2. **Détection auto des passages** : on suit la ligne la plus forte et on fitte
   le modèle Doppler exact (temps retardé). On en tire la vitesse, la distance
   au plus proche passage (CPA) et l'instant du CPA.
3. **ADS-B** : l'archive journalière adsb.lol (~3 Go) est lue en streaming et
   filtrée à la volée sur la zone. Seuls quelques Mo sont gardés en cache.
4. **Recherche grossière** (grille de 200 m) : on garde les points qui ont
   entendu un avion à hélice de la bonne vitesse, à la bonne distance, puis le
   suivant avec le bon écart de temps, sans aucun autre avion proche entre les deux.
5. **Score fin** (grille de 100 m) : pour chaque point, on modélise la courbe
   Doppler exacte produite par la vraie trajectoire de l'avion, puis on la
   compare au spectrogramme avec l'écart de temps verrouillé.
6. **Altitude du sol** : tuiles SRTM téléchargées automatiquement (AWS skadi).

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

## Limites

- Il faut connaître au moins la **date** (les métadonnées de la vidéo suffisent).
- Seuls les **avions à hélice** sont exploités. Les jets sont trop hauts et leur bruit est large bande.
- Couverture ADS-B : les données adsb.lol sont collectées par des bénévoles et peuvent être trouées dans les zones rurales.
- À utiliser sur tes propres enregistrements ou dans un cadre d'enquête légitime.
