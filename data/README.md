# Data

All data used in this research is **100% real-world data** from verified public sources.

## 6 Domains

| Domain | Source | Records | Verified |
|--------|--------|---------|----------|
| **Crypto** | Bybit Exchange | 44,000+ | ✅ Real |
| **Commodities** | FRED (US Government) | 5,630 | ✅ Real |
| **Weather** | Open-Meteo API | 9,105 | ✅ Real |
| **Solar** | Open-Meteo Solar API | 116,834 | ✅ Real |
| **Traffic** | NYC TLC Yellow Taxi | 2,879 | ✅ Real |
| **Air Quality** | Open-Meteo PM2.5 | 2,880 | ✅ Real |

## Data Sources

### Crypto (Bybit)
- **URL**: https://bybit.com
- **Data**: OHLCV price data for BTC, ETH, SOL, DOGE, XRP
- **Period**: 2021-2024
- **Files**: `bybit/*.csv`

### Commodities (FRED)
- **URL**: https://fred.stlouisfed.org
- **Data**: Daily prices for WTI Oil, Copper, Natural Gas
- **Period**: 2015-2024
- **Files**: `commodities/fred_real_prices.csv`

### Weather (Open-Meteo)
- **URL**: https://archive-api.open-meteo.com
- **Data**: Daily temperature, precipitation, wind for 5 US cities
- **Period**: 2021-2024
- **Files**: `weather/openmeteo_real_weather.csv`

### Solar (Open-Meteo)
- **URL**: https://archive-api.open-meteo.com
- **Data**: Hourly GHI, DNI, DHI irradiance for 5 US locations
- **Period**: 2023
- **Files**: `solar/openmeteo_real_solar.csv`

### Traffic (NYC TLC)
- **URL**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Data**: Yellow taxi trip counts (hourly aggregated)
- **Period**: Jan-Apr 2023
- **Files**: `traffic/nyc_taxi_real_hourly.csv`

### Air Quality (Open-Meteo)
- **URL**: https://open-meteo.com/en/docs/air-quality-api
- **Data**: PM2.5 concentrations for NYC
- **Period**: Jan-Apr 2023
- **Files**: `air_quality/openmeteo_real_air_quality.csv`

## Downloading Data

To re-download all data:

```bash
# From repository root
python scripts/collect_bybit_data.py           # Crypto
python scripts/download_fred_commodities_real.py  # Commodities
python scripts/download_real_weather.py        # Weather
python scripts/download_real_solar.py          # Solar
python scripts/download_real_traffic.py        # Traffic (NYC TLC, Jan-Apr 2023)
# Air Quality: See Open-Meteo air-quality API
```

## Data Verification

All data has been verified against original sources. See `REAL_DATA_MANIFEST.md` for detailed verification.
