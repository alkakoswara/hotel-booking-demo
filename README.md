# Hotel Booking Cancellation: H-1 Scoring Demo

A data science project on predicting hotel booking cancellations one day before arrival, built end to end as a personal learning project. It includes a live scoring demo and a notebook that documents the full reasoning, from the first business question to the final model decision.

**Live demo:** https://hotel-booking-demo-nuublj7myce5dnatebttq8.streamlit.app/

The demo shows the raw model score for one booking. That score is not a calibrated probability of cancellation, and the model is still in development. Treat the demo as a way to see how the model behaves, not as a booking decision tool.

## What is in this repository

| File | Purpose |
|---|---|
| `app_demo_d.py` | Streamlit demo app (the live demo above) |
| `preprocessing_d_demo.joblib` | Preprocessing step of the demo model (one-hot + passthrough) |
| `model_d_demo.ubj` | The demo model (XGBoost, official save format) |
| `requirements.txt` | Pinned packages for running the demo app |
| `Dockerfile` | Recipe to build a Docker image that runs the demo app |
| `hotel_booking_cancellation_h1_final.ipynb` | Project notebook: reasoning, analysis, results, and how the demo model was rebuilt and verified |

## The scenario

Imagine a hotel team that wants to spot, one day before arrival (H-1), which bookings are most likely to cancel, so staff can call and confirm the highest-risk ones. That scenario drives every choice in the project. The data is the City Hotel subset of the Hotel Booking Demand dataset (Antonio, de Almeida, Nunes 2019, Data in Brief 22:41-49): 79,330 bookings.

The cohort keeps only bookings that already existed at H-1 and had not cancelled by then: 42,859 bookings with 1,099 positive cases, about 2.6%: bookings that later cancelled or were recorded as no-shows (784 no-shows and 315 cancellations). The split follows time, not random shuffling:

- **train**: 27,697 bookings (790 positives), status known before 31 Dec 2016
- **validation**: 6,257 bookings (156 positives), used for model choice and the daily simulation
- **test**: 8,399 bookings (150 positives), the latest period
- **506 set aside** (238 from the train side, 268 from the validation side) because their status date falls outside the scoring window; 3 positives sit in these rows

A random split would mix past and future, so validation would no longer test a later period and could give an overoptimistic estimate for future use.

## What the project tried

- SQL and EDA to audit the data and answer the first business questions.
- A baseline model ("S9") with 4 features, which taught the main lesson early: 97.5% accuracy means nothing here, because a model that never predicts cancellation already scores 97.5%. From then on, ranking metrics (average precision) replace accuracy.
- Feature experiments (candidates A to E) and time-fold tuning of logistic regression, XGBoost, and CatBoost on the 12-feature recipe.
- A daily simulation of the scenario: each day, pick the top 5 or top 10 highest-scored bookings for a confirmation call.

## Results, stated plainly

- The demo model (candidate "D", 12 features, XGBoost with default settings) reaches AP 0.6654 on train and 0.1082 on validation. On validation, where positives are about 2.5% of bookings, a random ranking has an expected AP near that prevalence, so 0.1082 sits clearly above the random reference. On the later test period the model scores 0.0230, but see the limitations: that number is exploratory, not an independent final test.
- In the daily simulation on validation (118 days), the tuned XGBoost winner checks 590 bookings at 5 per day and catches 27 of 156 positives, at about 4% precision. Random picking with the same daily effort would be expected to catch about 13.5. At 10 per day: 47 caught versus an expected 27.0. The 27 and 47 come from one way of breaking tied scores at the daily cutoff; other tie-breakings give anywhere from 22 to 27 at 5 per day and 35 to 47 at 10 per day. Those ranges are analytical possibilities from the same scores, not a confidence interval. This is a descriptive comparison on one validation subset, not a significance test, and "caught" means the booking was later cancelled or no-showed, not that a cancellation was prevented. Most positives are still missed, and precision stays low.
- The tuned winners were picked by time-fold AP inside train (logistic regression C=0.01, XGBoost depth 3, CatBoost depth 6), but their single validation runs did not beat the untuned candidate D (0.0293, 0.0884, 0.0660 against 0.1082). A controlled depth cut (6 to 3) on a clone of D did not improve validation either. The cause of the weak generalization is not established; the project documents what was tried instead of claiming a fix.

## Why this model is in the demo

The demo uses candidate D because it is the model most discussed in the project's history and its saved artifact already existed, not because it scored best. Recipe C actually scored slightly higher on validation (0.1098 against 0.1082), and the daily simulation results above come from the tuned XGBoost and CatBoost winners, not from D. Part 16 of the notebook lays out this decision and its trade-offs, and Part 17 shows how D was rebuilt inside the notebook and verified: its AP on train and validation match the recorded numbers exactly.

The model is stored as two files (preprocessing as joblib, model in XGBoost's UBJ format) after a full-pipeline snapshot proved fragile across environments. Before export, all 6,257 validation scores are verified identical between the stored files and the in-memory pipeline.

## Limitations

- The model is not ready for hotel operations. The demo exists to explain how the model works and where it falls short.
- The test set was already evaluated once mid-project, so every number that came after, including the 0.0230 test AP above, is exploratory rather than an independent final measurement.
- Two guest-history features are covered by the paper's general H-1 snapshot methodology, but their per-variable description does not restate the snapshot date, so whether their values fully respect H-1 is not confirmed.
- Findings about cancellations are descriptive, not causal. The effect of a confirmation call on cancellations was never measured.

## Running the demo locally

```
pip install -r requirements.txt
streamlit run app_demo_d.py
```

`requirements.txt` covers only what the app needs to score a booking (inference), and was tested on Python 3.13.

## Running the demo with Docker

With Docker Desktop running (Linux containers), from the folder containing these files:

```
docker build -t hotel-booking-demo .
docker run --rm -p 127.0.0.1:8501:8501 hotel-booking-demo
```

Then open http://localhost:8501 in a browser. The `-p 127.0.0.1:8501:8501` mapping exposes the app only to your own machine. Press Ctrl+C to stop the app; the `--rm` flag removes the container afterwards, while the image stays on disk for the next run.

Tested on Windows 11 with Docker Desktop (WSL 2 backend). The container scored the example booking at 23.2%, the same number as the local and hosted runs; that was a single-input smoke test, not a full parity check across environments.

## Running the notebook

The notebook runs on Google Colab. Its recorded environment is Python 3.13, scikit-learn 1.6.1, XGBoost 3.4.1, and CatBoost 1.2.10. Run the install cell first, restart the runtime once, then run the rest in order. The notebook installs its own packages (including CatBoost, which the demo app does not need), so its setup is separate from `requirements.txt`. Tables marked as recorded results are copied from the actual evaluation sessions; not every experiment is rerun inside the notebook.

## Data source

- Article: https://www.sciencedirect.com/science/article/pii/S2352340918315191
- Official data (article supplement): https://ars.els-cdn.com/content/image/1-s2.0-S2352340918315191-mmc2.zip
- Mirror loaded by the notebook: https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv (verified identical to the official H2 file for City Hotel after format normalization)
