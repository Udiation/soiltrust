# Connectivity and public-data acquisition log

Checked 2026-09-03 from `<gpu-node>`. A site is marked reachable when DNS resolution and an HTTPS response both succeeded; this does not imply that its datasets are anonymously downloadable.

| Host | Reachable | Evidence |
|---|---:|---|
| `github.com` | YES | DNS resolved; HTTPS `200` |
| `raw.githubusercontent.com` | YES | DNS resolved; HTTPS endpoint responded (`301` at `/`, then `200` on GitHub) |
| `data.gov.in` | YES | HTTPS `200`, redirecting to `www.data.gov.in` |
| `soilhealth.dac.gov.in` | YES | HTTPS `200`; portal SPA and its 4,065,058-byte JavaScript bundle downloaded |
| `agcensus.gov.in` | YES | HTTPS `200`; `/AgriCensus/` returned a 64,851-byte application page |
| `bhoonidhi.nrsc.gov.in` | YES | HTTPS `200` after redirect to `/bhoonidhi/home.html` |

## Downloads and attempts

- Boundary source: [`yashveeeeeeer/india-geodata`](https://github.com/yashveeeeeeer/india-geodata), commit `6e5a00c781a6d50e580bd8752526293ef18a54da`.
- District: its Census 2011 district shapefile, filtered by `ST_NM = Tamil Nadu` and `DISTRICT = Cuddalore`.
- Villages: its `admin/villages` release asset `LGD_Villages.parquet`, filtered by `stname = TAMIL NADU` and `dtname = Cuddalore`. The upstream metadata attributes this layer to LGD/SOI/Bhuvan and marks it CC0.
- `data.gov.in`: an anonymous search for `soil health Cuddalore` returned the portal shell (`200`) but no directly downloadable Cuddalore nutrient aggregate was exposed in the returned page.
- Soil Health Card portal: the public application and GraphQL client are reachable. No anonymous, documented export of Cuddalore sample-level or district aggregate nutrient data was identified. No credentials were supplied and no restricted endpoint was bypassed.
- Agriculture Census portal: the public application is reachable. A guessed public table URL returned `404`; no directly downloadable Cuddalore land-holding extract was identified without using the portal's interactive reporting workflow.

Therefore the government layers are **not included**. “Reachable” and “data acquired” are intentionally distinguished.
