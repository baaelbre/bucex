UCCLE DAILY TEMPERATURE UPDATE THROUGH 31 AUGUST 2026

Data file: Uccle_31_08_26.csv
Original file: Uccle_24_10_23(1).csv
Original observations: 1892-01-01 through 2023-10-17 (48,137 rows).
Added dates: 2023-10-18 through 2026-08-31 (1,049 rows).
Total: 49,186 daily rows. Dates are continuous and unique.
Columns retained: DAY (YYYY-MM-DD), TX (maximum temperature, degrees Celsius), TN (minimum temperature, degrees Celsius), RR (precipitation).
All original bytes and observations are unchanged. New RR entries are NA because this was a temperature-only update.

SOURCE AND EXTRACTION
Station: Uccle, Belgium, Meteociel code 6447, altitude 100 m.
Retrieved: 2026-09-14.
2023-10-18 through 2023-12-31: Meteociel.
2024-01-01 through 2024-12-31: user-supplied Meteostat export.xlsx, with one exception on 2024-12-11 described below.
2025-01-01 through 2026-08-31: Meteociel.
Meteostat mapping: date -> DAY (date component), tmax -> TX, tmin -> TN. Values were used in degrees Celsius as supplied. The export contains 366 dates, and 365 nonmissing TX and TN values.
Meteostat export.xlsx does not embed a station identifier or daily observation-window metadata. It was supplied by the user for the Uccle 2024 update. No claim of matching day definitions across the sources is made.
Meteostat has neither TX nor TN on 2024-12-11. Those two fields were filled from the Meteociel monthly table (TX=3.9, TN=4.0); this pair is explicitly flagged below.
Monthly URL pattern: https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=M&annee=YYYY&sn=0
Here M is the calendar month (1 to 12). The supplied meteociel.com domain and meteociel.fr displayed identical May 2024 observations in a comparison.

Reporting windows stated by Meteociel:
TX: 06:00 UTC on DAY through 06:00 UTC the following day.
TN: 18:00 UTC on the preceding day through 18:00 UTC on DAY.
The daily detail URL uses a zero-based month: https://www.meteociel.fr/temps-reel/obs_villes.php?code2=6447&jour2=D&mois2=M_MINUS_1&annee2=YYYY

METEOCIEL MONTHLY-TABLE GAPS AND DAILY-SUMMARY FALLBACKS
The downloaded monthly tables lacked 173 TX values and 14 TN values.
The Meteociel monthly TX series has a continuous gap from 2024-02-12 through 2024-07-28. The user-supplied Meteostat export resolves this period; Meteociel daily summaries were not used for 2024.
Values recovered from the headline summaries on Meteociel's daily pages: TX=3, TN=5.
Outside 2024, only missing monthly entries were filled from daily summaries. Other monthly-table values were retained. For 2024 the supplied Meteostat observations take precedence.
The daily-page summaries carry the same reporting-window labels. Values were transcribed as reported, with no interpolation, rounding beyond the displayed precision, or manual calculation from hourly observations.
However, the website does not establish for every fallback whether the value is a continuously measured extremum or derived from discrete observations. The fallback dates below should be distinguishable in analyses sensitive to tail measurement.
TX daily-summary fallback dates: 2023-11-09; 2023-11-28; 2025-10-22.
TN daily-summary fallback dates: 2023-10-26; 2023-11-10; 2023-11-29; 2023-12-05; 2025-10-23.

REMAINING MISSING VALUES
TX: 0. Dates: none.
TN: 0. Dates: none.
Unavailable temperatures are represented as NA. Missing observations are not zeros.

COMPARISON WITH THE ORIGINAL SERIES
The overlapping dates 2023-10-01 through 2023-10-17 were compared before appending.
8 of those 17 dates have at least one differing nonmissing temperature. Original values were retained.
The overlap is insufficient to establish a homogeneous measurement convention or to identify the cause of discrepancies. No date shift or statistical adjustment was applied.
Example: 2023-10-03 has original TX=22.7 and TN=12.0; the current Meteociel monthly table reports TX=18.2 and TN=17.9.
Overlap comparison (degrees Celsius):
DAY,original_TX,meteociel_TX,original_TN,meteociel_TN
2023-10-01,24.2,24.2,11.4,11.4
2023-10-02,26.5,26.5,14.7,14.7
2023-10-03,22.7,18.2,12,17.9
2023-10-04,17.9,17.9,10.4,10.4
2023-10-05,17.8,17.8,11.1,12.1
2023-10-06,19.8,19.8,10.8,10.8
2023-10-07,22,22,13.2,13.2
2023-10-08,22.7,22.7,12.5,12.5
2023-10-09,20.2,20.2,13.4,14.6
2023-10-10,24.7,24.7,12,12
2023-10-11,22.5,22.5,15.6,15.5
2023-10-12,17.8,17.3,14.3,14.6
2023-10-13,22.4,22.4,16,17.3
2023-10-14,16,13.2,6.2,8.9
2023-10-15,12.1,NA,3.9,5.7
2023-10-16,12.1,NA,3.2,3.2
2023-10-17,13.1,13.1,7.1,NA

REPORTED TN GREATER THAN TX
The following source-reported pairs were retained. TX and TN have different observation windows; these are not extrema of the same 24-hour interval. Source quality may also contribute, so review these dates for joint or ordered analyses.
DAY,TX,TN
2023-11-30,2.2,2.3
2024-12-11,3.9,4

VALIDATION
Checked station identity, requested month/year, calendar day and weekday, column labels, duplicate dates, continuous date coverage, numeric temperatures, and consistency with the temperature graph parameters wherever present.
Validated each added output cell against the extracted source record and confirmed the original file is an unchanged byte prefix of the updated CSV.
This extension combines the user-supplied historical series, the supplied Meteostat export for 2024, and Meteociel. No homogenization or independent quality correction was performed. For the SERRA extremes analysis, assess measurement comparability across source boundaries and the flagged dates.

DAILY SUMMARY VALUES USED (for exact provenance)
DAY,filled_columns,TX,TN,source_url
2023-10-26,TN,,6.8,https://www.meteociel.fr/temps-reel/obs_villes.php?code2=6447&jour2=26&mois2=9&annee2=2023
2023-11-09,TX,11.1,,https://www.meteociel.fr/temps-reel/obs_villes.php?code2=6447&jour2=9&mois2=10&annee2=2023
2023-11-10,TN,,6.6,https://www.meteociel.fr/temps-reel/obs_villes.php?code2=6447&jour2=10&mois2=10&annee2=2023
2023-11-28,TX,2.2,,https://www.meteociel.fr/temps-reel/obs_villes.php?code2=6447&jour2=28&mois2=10&annee2=2023
2023-11-29,TN,,1.9,https://www.meteociel.fr/temps-reel/obs_villes.php?code2=6447&jour2=29&mois2=10&annee2=2023
2023-12-05,TN,,3.7,https://www.meteociel.fr/temps-reel/obs_villes.php?code2=6447&jour2=5&mois2=11&annee2=2023
2025-10-22,TX,14.3,,https://www.meteociel.fr/temps-reel/obs_villes.php?code2=6447&jour2=22&mois2=9&annee2=2025
2025-10-23,TN,,9.6,https://www.meteociel.fr/temps-reel/obs_villes.php?code2=6447&jour2=23&mois2=9&annee2=2025

MONTHLY SOURCES
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=10&annee=2023&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=11&annee=2023&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=12&annee=2023&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=12&annee=2024&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=1&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=2&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=3&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=4&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=5&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=6&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=7&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=8&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=9&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=10&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=11&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=12&annee=2025&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=1&annee=2026&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=2&annee=2026&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=3&annee=2026&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=4&annee=2026&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=5&annee=2026&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=6&annee=2026&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=7&annee=2026&sn=0
https://www.meteociel.fr/climatologie/obs_villes.php?code2=6447&mois=8&annee=2026&sn=0
