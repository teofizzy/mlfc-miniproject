
# Regional Climate Warming in Africa – Mini Project
## 📌 Overview

This project analyzes temperature trends across major African regions using monthly climate data (1979–2024). The goal is to identify which regions are warming fastest, quantify variability, and assess recent anomalies relative to a standard baseline (1991–2020).

The results highlight regional disparities in warming rates, offering insights for climate science and adaptation planning.

## 🌍 Regions Analyzed

Bounding boxes were used to extract data for the following representative regions:

Sahel

East Africa

Central Africa

Congo Basin

Southern Africa

Sahara

## 📊 Methods

### Data Input:

- Monthly mean 2-meter temperature (1979–2024).

- Baseline period: 1991–2020.

- Regional Summaries: For each region, the following metrics were computed:

- Trend (°C/decade) – Linear regression over time.

- Mean Temperature (°C) – Long-term mean.

- Variability (StdDev °C) – Interannual standard deviation.

- Recent Anomaly (°C) – Mean anomaly (2014–2023) relative to 1991–2020.

## 🛠️ Key Functions

`compute_regional_mean(da)` → extracts spatial mean for a region.

`compute_anomalies(da, baseline)` → calculates anomalies relative to baseline.

`summarize_region(da, name)` → returns summary stats (trend, mean, std, anomaly).

## 🔑 Findings

- All regions show warming since 1979, with trends ranging 0.2–0.4 °C/decade.

- Sahara and Congo Basin warm the fastest.

- Southern Africa warms more slowly, though still significant.

- The last decade (2014–2023) is 0.3–0.6 °C warmer than the 1991–2020 baseline.

### Fynesse template
This miniproject uses the fynesse structure.
The Fynesse paradigm considers three aspects to data analysis, Access, Assess, Address.
