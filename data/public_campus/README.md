# Public campus inputs

Built by `yanmar.public_campus_profiles.public_whr_profiles()`.

## Sources

- UKPN standard profiles: industrial → manufacturing; commercial → warehouse/refrigeration diurnal envelope
- Cambridge Estates Building Energy Archive (b0, 2019) → office shape
- PVGIS v5.3 Cambridge 2019 → PV shape + T2m; annual PV energy scaled to CASE CF (DESNZ-anchored)
- CIBSE TM41 degree-day bases for HVAC split; TM46 intensity context in the manuscript
- Yanmar SE220-100 500–800 °C box for deterministic Ts(h)

No random seed. Re-fetch with `force_refresh=True`.
