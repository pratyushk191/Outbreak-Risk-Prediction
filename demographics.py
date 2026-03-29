"""
demographics.py
---------------
Embedded country-level demographic and healthcare capacity data.
Sources: World Bank, UN Population Division, WHO (2019–2022 estimates).

Columns
-------
location                  : country name (matches OWID/JHU naming)
population_density        : people per km²
median_age                : years
urban_population_pct      : % of population in urban areas
hospital_beds_per_thousand: beds per 1 000 inhabitants
gdp_per_capita_usd        : USD (current), proxy for healthcare resources
"""

from __future__ import annotations

import pandas as pd


_DEMOGRAPHIC_DATA: list[dict] = [
    # location, pop_density, median_age, urban_pct, hosp_beds, gdp_pc
    {"location": "Afghanistan", "population_density": 60.0, "median_age": 18.4, "urban_population_pct": 26.0, "hospital_beds_per_thousand": 0.5, "gdp_per_capita_usd": 500},
    {"location": "Albania", "population_density": 105.0, "median_age": 38.0, "urban_population_pct": 63.0, "hospital_beds_per_thousand": 2.9, "gdp_per_capita_usd": 5200},
    {"location": "Algeria", "population_density": 18.0, "median_age": 29.1, "urban_population_pct": 74.0, "hospital_beds_per_thousand": 1.9, "gdp_per_capita_usd": 3600},
    {"location": "Angola", "population_density": 26.0, "median_age": 16.7, "urban_population_pct": 67.0, "hospital_beds_per_thousand": 0.8, "gdp_per_capita_usd": 1900},
    {"location": "Argentina", "population_density": 16.0, "median_age": 31.7, "urban_population_pct": 92.0, "hospital_beds_per_thousand": 5.0, "gdp_per_capita_usd": 10600},
    {"location": "Armenia", "population_density": 105.0, "median_age": 35.6, "urban_population_pct": 63.0, "hospital_beds_per_thousand": 4.2, "gdp_per_capita_usd": 4200},
    {"location": "Australia", "population_density": 3.0, "median_age": 37.9, "urban_population_pct": 86.0, "hospital_beds_per_thousand": 3.8, "gdp_per_capita_usd": 54000},
    {"location": "Austria", "population_density": 109.0, "median_age": 44.3, "urban_population_pct": 59.0, "hospital_beds_per_thousand": 7.4, "gdp_per_capita_usd": 48000},
    {"location": "Azerbaijan", "population_density": 121.0, "median_age": 32.7, "urban_population_pct": 57.0, "hospital_beds_per_thousand": 4.7, "gdp_per_capita_usd": 4500},
    {"location": "Bahrain", "population_density": 2239.0, "median_age": 32.3, "urban_population_pct": 89.0, "hospital_beds_per_thousand": 1.7, "gdp_per_capita_usd": 24500},
    {"location": "Bangladesh", "population_density": 1265.0, "median_age": 27.6, "urban_population_pct": 39.0, "hospital_beds_per_thousand": 0.8, "gdp_per_capita_usd": 1850},
    {"location": "Belarus", "population_density": 47.0, "median_age": 40.3, "urban_population_pct": 80.0, "hospital_beds_per_thousand": 10.8, "gdp_per_capita_usd": 6300},
    {"location": "Belgium", "population_density": 383.0, "median_age": 41.8, "urban_population_pct": 98.0, "hospital_beds_per_thousand": 5.6, "gdp_per_capita_usd": 45000},
    {"location": "Bolivia", "population_density": 10.0, "median_age": 25.5, "urban_population_pct": 70.0, "hospital_beds_per_thousand": 1.3, "gdp_per_capita_usd": 3200},
    {"location": "Bosnia and Herzegovina", "population_density": 64.0, "median_age": 42.3, "urban_population_pct": 50.0, "hospital_beds_per_thousand": 3.5, "gdp_per_capita_usd": 6000},
    {"location": "Brazil", "population_density": 25.0, "median_age": 33.5, "urban_population_pct": 87.0, "hospital_beds_per_thousand": 2.1, "gdp_per_capita_usd": 8700},
    {"location": "Bulgaria", "population_density": 62.0, "median_age": 44.6, "urban_population_pct": 76.0, "hospital_beds_per_thousand": 7.5, "gdp_per_capita_usd": 9900},
    {"location": "Cambodia", "population_density": 95.0, "median_age": 26.4, "urban_population_pct": 24.0, "hospital_beds_per_thousand": 0.8, "gdp_per_capita_usd": 1500},
    {"location": "Cameroon", "population_density": 50.0, "median_age": 18.5, "urban_population_pct": 57.0, "hospital_beds_per_thousand": 1.3, "gdp_per_capita_usd": 1400},
    {"location": "Canada", "population_density": 4.0, "median_age": 41.4, "urban_population_pct": 82.0, "hospital_beds_per_thousand": 2.5, "gdp_per_capita_usd": 43000},
    {"location": "Chile", "population_density": 26.0, "median_age": 35.4, "urban_population_pct": 88.0, "hospital_beds_per_thousand": 2.1, "gdp_per_capita_usd": 13500},
    {"location": "China", "population_density": 148.0, "median_age": 38.4, "urban_population_pct": 64.0, "hospital_beds_per_thousand": 4.3, "gdp_per_capita_usd": 10500},
    {"location": "Colombia", "population_density": 44.0, "median_age": 30.9, "urban_population_pct": 81.0, "hospital_beds_per_thousand": 1.7, "gdp_per_capita_usd": 6300},
    {"location": "Costa Rica", "population_density": 100.0, "median_age": 32.6, "urban_population_pct": 82.0, "hospital_beds_per_thousand": 1.1, "gdp_per_capita_usd": 11800},
    {"location": "Croatia", "population_density": 73.0, "median_age": 43.9, "urban_population_pct": 58.0, "hospital_beds_per_thousand": 5.5, "gdp_per_capita_usd": 14700},
    {"location": "Cuba", "population_density": 112.0, "median_age": 42.1, "urban_population_pct": 77.0, "hospital_beds_per_thousand": 5.3, "gdp_per_capita_usd": 8800},
    {"location": "Cyprus", "population_density": 130.0, "median_age": 37.3, "urban_population_pct": 67.0, "hospital_beds_per_thousand": 3.4, "gdp_per_capita_usd": 27000},
    {"location": "Czechia", "population_density": 139.0, "median_age": 43.2, "urban_population_pct": 74.0, "hospital_beds_per_thousand": 6.6, "gdp_per_capita_usd": 22500},
    {"location": "Democratic Republic of Congo", "population_density": 40.0, "median_age": 17.0, "urban_population_pct": 46.0, "hospital_beds_per_thousand": 0.8, "gdp_per_capita_usd": 550},
    {"location": "Denmark", "population_density": 137.0, "median_age": 42.0, "urban_population_pct": 88.0, "hospital_beds_per_thousand": 2.6, "gdp_per_capita_usd": 60000},
    {"location": "Dominican Republic", "population_density": 228.0, "median_age": 28.1, "urban_population_pct": 83.0, "hospital_beds_per_thousand": 1.6, "gdp_per_capita_usd": 7700},
    {"location": "Ecuador", "population_density": 68.0, "median_age": 28.8, "urban_population_pct": 64.0, "hospital_beds_per_thousand": 1.5, "gdp_per_capita_usd": 5800},
    {"location": "Egypt", "population_density": 100.0, "median_age": 24.6, "urban_population_pct": 43.0, "hospital_beds_per_thousand": 1.6, "gdp_per_capita_usd": 3500},
    {"location": "Estonia", "population_density": 31.0, "median_age": 42.7, "urban_population_pct": 70.0, "hospital_beds_per_thousand": 4.7, "gdp_per_capita_usd": 23000},
    {"location": "Ethiopia", "population_density": 115.0, "median_age": 19.5, "urban_population_pct": 22.0, "hospital_beds_per_thousand": 0.3, "gdp_per_capita_usd": 850},
    {"location": "Finland", "population_density": 18.0, "median_age": 42.8, "urban_population_pct": 86.0, "hospital_beds_per_thousand": 3.6, "gdp_per_capita_usd": 48000},
    {"location": "France", "population_density": 119.0, "median_age": 41.7, "urban_population_pct": 81.0, "hospital_beds_per_thousand": 5.9, "gdp_per_capita_usd": 40000},
    {"location": "Germany", "population_density": 240.0, "median_age": 45.7, "urban_population_pct": 77.0, "hospital_beds_per_thousand": 8.0, "gdp_per_capita_usd": 46000},
    {"location": "Ghana", "population_density": 131.0, "median_age": 21.4, "urban_population_pct": 57.0, "hospital_beds_per_thousand": 0.9, "gdp_per_capita_usd": 2200},
    {"location": "Greece", "population_density": 83.0, "median_age": 45.3, "urban_population_pct": 80.0, "hospital_beds_per_thousand": 4.2, "gdp_per_capita_usd": 18000},
    {"location": "Guatemala", "population_density": 166.0, "median_age": 22.9, "urban_population_pct": 52.0, "hospital_beds_per_thousand": 0.4, "gdp_per_capita_usd": 4500},
    {"location": "Honduras", "population_density": 89.0, "median_age": 24.2, "urban_population_pct": 58.0, "hospital_beds_per_thousand": 0.7, "gdp_per_capita_usd": 2400},
    {"location": "Hungary", "population_density": 108.0, "median_age": 43.4, "urban_population_pct": 72.0, "hospital_beds_per_thousand": 7.0, "gdp_per_capita_usd": 16500},
    {"location": "India", "population_density": 464.0, "median_age": 28.4, "urban_population_pct": 35.0, "hospital_beds_per_thousand": 0.5, "gdp_per_capita_usd": 2100},
    {"location": "Indonesia", "population_density": 151.0, "median_age": 29.7, "urban_population_pct": 57.0, "hospital_beds_per_thousand": 1.0, "gdp_per_capita_usd": 4100},
    {"location": "Iran", "population_density": 52.0, "median_age": 32.4, "urban_population_pct": 76.0, "hospital_beds_per_thousand": 1.5, "gdp_per_capita_usd": 5500},
    {"location": "Iraq", "population_density": 93.0, "median_age": 20.9, "urban_population_pct": 71.0, "hospital_beds_per_thousand": 1.3, "gdp_per_capita_usd": 5100},
    {"location": "Ireland", "population_density": 72.0, "median_age": 37.8, "urban_population_pct": 64.0, "hospital_beds_per_thousand": 3.0, "gdp_per_capita_usd": 78000},
    {"location": "Israel", "population_density": 416.0, "median_age": 30.4, "urban_population_pct": 93.0, "hospital_beds_per_thousand": 3.0, "gdp_per_capita_usd": 43000},
    {"location": "Italy", "population_density": 206.0, "median_age": 47.3, "urban_population_pct": 71.0, "hospital_beds_per_thousand": 3.2, "gdp_per_capita_usd": 32000},
    {"location": "Japan", "population_density": 347.0, "median_age": 48.4, "urban_population_pct": 92.0, "hospital_beds_per_thousand": 13.1, "gdp_per_capita_usd": 40000},
    {"location": "Jordan", "population_density": 115.0, "median_age": 23.4, "urban_population_pct": 91.0, "hospital_beds_per_thousand": 1.4, "gdp_per_capita_usd": 4200},
    {"location": "Kazakhstan", "population_density": 7.0, "median_age": 31.6, "urban_population_pct": 58.0, "hospital_beds_per_thousand": 6.7, "gdp_per_capita_usd": 9000},
    {"location": "Kenya", "population_density": 94.0, "median_age": 20.1, "urban_population_pct": 28.0, "hospital_beds_per_thousand": 1.4, "gdp_per_capita_usd": 1900},
    {"location": "Kuwait", "population_density": 232.0, "median_age": 33.7, "urban_population_pct": 100.0, "hospital_beds_per_thousand": 2.0, "gdp_per_capita_usd": 31000},
    {"location": "Latvia", "population_density": 31.0, "median_age": 43.7, "urban_population_pct": 68.0, "hospital_beds_per_thousand": 5.6, "gdp_per_capita_usd": 17500},
    {"location": "Lebanon", "population_density": 667.0, "median_age": 30.8, "urban_population_pct": 89.0, "hospital_beds_per_thousand": 2.7, "gdp_per_capita_usd": 7700},
    {"location": "Libya", "population_density": 4.0, "median_age": 29.0, "urban_population_pct": 81.0, "hospital_beds_per_thousand": 3.7, "gdp_per_capita_usd": 7500},
    {"location": "Lithuania", "population_density": 44.0, "median_age": 43.8, "urban_population_pct": 68.0, "hospital_beds_per_thousand": 6.5, "gdp_per_capita_usd": 19500},
    {"location": "Luxembourg", "population_density": 242.0, "median_age": 39.3, "urban_population_pct": 91.0, "hospital_beds_per_thousand": 4.3, "gdp_per_capita_usd": 113000},
    {"location": "Malaysia", "population_density": 99.0, "median_age": 30.3, "urban_population_pct": 77.0, "hospital_beds_per_thousand": 1.9, "gdp_per_capita_usd": 11100},
    {"location": "Mexico", "population_density": 66.0, "median_age": 29.3, "urban_population_pct": 81.0, "hospital_beds_per_thousand": 1.0, "gdp_per_capita_usd": 9900},
    {"location": "Moldova", "population_density": 122.0, "median_age": 37.7, "urban_population_pct": 43.0, "hospital_beds_per_thousand": 5.8, "gdp_per_capita_usd": 3300},
    {"location": "Morocco", "population_density": 84.0, "median_age": 29.3, "urban_population_pct": 64.0, "hospital_beds_per_thousand": 1.0, "gdp_per_capita_usd": 3140},
    {"location": "Mozambique", "population_density": 40.0, "median_age": 17.4, "urban_population_pct": 38.0, "hospital_beds_per_thousand": 0.7, "gdp_per_capita_usd": 500},
    {"location": "Myanmar", "population_density": 83.0, "median_age": 28.2, "urban_population_pct": 31.0, "hospital_beds_per_thousand": 0.9, "gdp_per_capita_usd": 1400},
    {"location": "Nepal", "population_density": 204.0, "median_age": 24.6, "urban_population_pct": 21.0, "hospital_beds_per_thousand": 0.3, "gdp_per_capita_usd": 1000},
    {"location": "Netherlands", "population_density": 508.0, "median_age": 42.8, "urban_population_pct": 93.0, "hospital_beds_per_thousand": 3.2, "gdp_per_capita_usd": 52000},
    {"location": "New Zealand", "population_density": 19.0, "median_age": 37.9, "urban_population_pct": 87.0, "hospital_beds_per_thousand": 2.6, "gdp_per_capita_usd": 41000},
    {"location": "Nigeria", "population_density": 226.0, "median_age": 18.1, "urban_population_pct": 53.0, "hospital_beds_per_thousand": 0.5, "gdp_per_capita_usd": 2100},
    {"location": "Norway", "population_density": 14.0, "median_age": 39.8, "urban_population_pct": 83.0, "hospital_beds_per_thousand": 3.6, "gdp_per_capita_usd": 82000},
    {"location": "Oman", "population_density": 16.0, "median_age": 30.6, "urban_population_pct": 86.0, "hospital_beds_per_thousand": 1.5, "gdp_per_capita_usd": 17300},
    {"location": "Pakistan", "population_density": 287.0, "median_age": 22.8, "urban_population_pct": 37.0, "hospital_beds_per_thousand": 0.6, "gdp_per_capita_usd": 1500},
    {"location": "Panama", "population_density": 57.0, "median_age": 30.0, "urban_population_pct": 69.0, "hospital_beds_per_thousand": 2.3, "gdp_per_capita_usd": 13700},
    {"location": "Paraguay", "population_density": 17.0, "median_age": 26.5, "urban_population_pct": 62.0, "hospital_beds_per_thousand": 0.8, "gdp_per_capita_usd": 5400},
    {"location": "Peru", "population_density": 26.0, "median_age": 29.1, "urban_population_pct": 78.0, "hospital_beds_per_thousand": 1.6, "gdp_per_capita_usd": 6700},
    {"location": "Philippines", "population_density": 368.0, "median_age": 25.7, "urban_population_pct": 47.0, "hospital_beds_per_thousand": 1.0, "gdp_per_capita_usd": 3500},
    {"location": "Poland", "population_density": 124.0, "median_age": 41.7, "urban_population_pct": 60.0, "hospital_beds_per_thousand": 6.5, "gdp_per_capita_usd": 15700},
    {"location": "Portugal", "population_density": 112.0, "median_age": 45.5, "urban_population_pct": 66.0, "hospital_beds_per_thousand": 3.5, "gdp_per_capita_usd": 22500},
    {"location": "Qatar", "population_density": 248.0, "median_age": 33.2, "urban_population_pct": 99.0, "hospital_beds_per_thousand": 1.2, "gdp_per_capita_usd": 58000},
    {"location": "Romania", "population_density": 84.0, "median_age": 42.5, "urban_population_pct": 55.0, "hospital_beds_per_thousand": 6.9, "gdp_per_capita_usd": 12600},
    {"location": "Russia", "population_density": 9.0, "median_age": 39.6, "urban_population_pct": 75.0, "hospital_beds_per_thousand": 8.1, "gdp_per_capita_usd": 11300},
    {"location": "Saudi Arabia", "population_density": 16.0, "median_age": 31.9, "urban_population_pct": 84.0, "hospital_beds_per_thousand": 2.2, "gdp_per_capita_usd": 22300},
    {"location": "Senegal", "population_density": 82.0, "median_age": 18.7, "urban_population_pct": 48.0, "hospital_beds_per_thousand": 0.3, "gdp_per_capita_usd": 1400},
    {"location": "Serbia", "population_density": 80.0, "median_age": 43.2, "urban_population_pct": 56.0, "hospital_beds_per_thousand": 5.6, "gdp_per_capita_usd": 7400},
    {"location": "Singapore", "population_density": 8358.0, "median_age": 42.2, "urban_population_pct": 100.0, "hospital_beds_per_thousand": 2.4, "gdp_per_capita_usd": 65000},
    {"location": "Slovakia", "population_density": 114.0, "median_age": 41.2, "urban_population_pct": 54.0, "hospital_beds_per_thousand": 5.8, "gdp_per_capita_usd": 19000},
    {"location": "Slovenia", "population_density": 103.0, "median_age": 44.5, "urban_population_pct": 55.0, "hospital_beds_per_thousand": 4.5, "gdp_per_capita_usd": 25500},
    {"location": "South Africa", "population_density": 49.0, "median_age": 27.6, "urban_population_pct": 67.0, "hospital_beds_per_thousand": 2.3, "gdp_per_capita_usd": 5900},
    {"location": "South Korea", "population_density": 527.0, "median_age": 43.7, "urban_population_pct": 82.0, "hospital_beds_per_thousand": 12.4, "gdp_per_capita_usd": 31500},
    {"location": "Spain", "population_density": 94.0, "median_age": 44.9, "urban_population_pct": 81.0, "hospital_beds_per_thousand": 3.0, "gdp_per_capita_usd": 29000},
    {"location": "Sri Lanka", "population_density": 341.0, "median_age": 33.7, "urban_population_pct": 19.0, "hospital_beds_per_thousand": 4.2, "gdp_per_capita_usd": 4000},
    {"location": "Sudan", "population_density": 25.0, "median_age": 19.8, "urban_population_pct": 35.0, "hospital_beds_per_thousand": 0.7, "gdp_per_capita_usd": 700},
    {"location": "Sweden", "population_density": 25.0, "median_age": 41.1, "urban_population_pct": 88.0, "hospital_beds_per_thousand": 2.2, "gdp_per_capita_usd": 52000},
    {"location": "Switzerland", "population_density": 219.0, "median_age": 43.1, "urban_population_pct": 74.0, "hospital_beds_per_thousand": 4.5, "gdp_per_capita_usd": 83000},
    {"location": "Syria", "population_density": 95.0, "median_age": 22.4, "urban_population_pct": 57.0, "hospital_beds_per_thousand": 1.5, "gdp_per_capita_usd": 1200},
    {"location": "Taiwan", "population_density": 673.0, "median_age": 42.3, "urban_population_pct": 79.0, "hospital_beds_per_thousand": 6.2, "gdp_per_capita_usd": 32000},
    {"location": "Tanzania", "population_density": 68.0, "median_age": 17.7, "urban_population_pct": 37.0, "hospital_beds_per_thousand": 0.7, "gdp_per_capita_usd": 1100},
    {"location": "Thailand", "population_density": 135.0, "median_age": 40.1, "urban_population_pct": 52.0, "hospital_beds_per_thousand": 2.1, "gdp_per_capita_usd": 7200},
    {"location": "Tunisia", "population_density": 76.0, "median_age": 32.7, "urban_population_pct": 70.0, "hospital_beds_per_thousand": 2.3, "gdp_per_capita_usd": 3400},
    {"location": "Turkey", "population_density": 110.0, "median_age": 32.4, "urban_population_pct": 77.0, "hospital_beds_per_thousand": 2.8, "gdp_per_capita_usd": 9700},
    {"location": "Uganda", "population_density": 214.0, "median_age": 16.4, "urban_population_pct": 26.0, "hospital_beds_per_thousand": 0.5, "gdp_per_capita_usd": 800},
    {"location": "Ukraine", "population_density": 77.0, "median_age": 41.4, "urban_population_pct": 70.0, "hospital_beds_per_thousand": 8.8, "gdp_per_capita_usd": 3700},
    {"location": "United Arab Emirates", "population_density": 116.0, "median_age": 33.5, "urban_population_pct": 87.0, "hospital_beds_per_thousand": 1.2, "gdp_per_capita_usd": 43000},
    {"location": "United Kingdom", "population_density": 281.0, "median_age": 40.5, "urban_population_pct": 84.0, "hospital_beds_per_thousand": 2.5, "gdp_per_capita_usd": 41000},
    {"location": "United States", "population_density": 36.0, "median_age": 38.3, "urban_population_pct": 83.0, "hospital_beds_per_thousand": 2.9, "gdp_per_capita_usd": 63000},
    {"location": "Uruguay", "population_density": 20.0, "median_age": 35.5, "urban_population_pct": 96.0, "hospital_beds_per_thousand": 2.4, "gdp_per_capita_usd": 16300},
    {"location": "Uzbekistan", "population_density": 76.0, "median_age": 28.2, "urban_population_pct": 51.0, "hospital_beds_per_thousand": 4.0, "gdp_per_capita_usd": 1800},
    {"location": "Venezuela", "population_density": 36.0, "median_age": 29.9, "urban_population_pct": 89.0, "hospital_beds_per_thousand": 0.8, "gdp_per_capita_usd": 3300},
    {"location": "Vietnam", "population_density": 314.0, "median_age": 31.9, "urban_population_pct": 38.0, "hospital_beds_per_thousand": 2.6, "gdp_per_capita_usd": 2700},
    {"location": "Yemen", "population_density": 54.0, "median_age": 19.6, "urban_population_pct": 39.0, "hospital_beds_per_thousand": 0.7, "gdp_per_capita_usd": 700},
    {"location": "Zambia", "population_density": 24.0, "median_age": 17.3, "urban_population_pct": 45.0, "hospital_beds_per_thousand": 2.0, "gdp_per_capita_usd": 1200},
    {"location": "Zimbabwe", "population_density": 38.0, "median_age": 19.0, "urban_population_pct": 32.0, "hospital_beds_per_thousand": 1.7, "gdp_per_capita_usd": 1200},
]


def load_demographics() -> pd.DataFrame:
    """Return a DataFrame of country demographic features.

    The returned DataFrame contains one row per country with columns:
    - location
    - population_density        (people / km²)
    - median_age                (years)
    - urban_population_pct      (%)
    - hospital_beds_per_thousand
    - gdp_per_capita_usd
    - healthcare_index           (derived: composite 0–1 score)
    - vulnerability_index        (derived: inversely weighted resource score)
    """
    df = pd.DataFrame(_DEMOGRAPHIC_DATA)

    # --- Derived features ---

    # Healthcare index: blend of beds and GDP (both normalised to [0,1])
    beds_norm = (df["hospital_beds_per_thousand"] - df["hospital_beds_per_thousand"].min()) / (
        df["hospital_beds_per_thousand"].max() - df["hospital_beds_per_thousand"].min()
    )
    gdp_norm = (df["gdp_per_capita_usd"] - df["gdp_per_capita_usd"].min()) / (
        df["gdp_per_capita_usd"].max() - df["gdp_per_capita_usd"].min()
    )
    df["healthcare_index"] = (0.5 * beds_norm + 0.5 * gdp_norm).round(4)

    # Vulnerability index: high density + young population + low healthcare = risky
    density_norm = (df["population_density"] - df["population_density"].min()) / (
        df["population_density"].max() - df["population_density"].min()
    )
    age_risk = 1 - (df["median_age"] - df["median_age"].min()) / (
        df["median_age"].max() - df["median_age"].min()
    )  # younger → higher spread risk
    df["vulnerability_index"] = (
        0.4 * density_norm + 0.3 * age_risk + 0.3 * (1 - gdp_norm)
    ).round(4)

    return df
