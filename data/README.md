# De-identified demonstration data

These files are supplied to demonstrate the model and its training workflow. They do not represent any particular basin, station, or calendar period.

| File | Rows | Time step |
| --- | ---: | --- |
| `series_a.csv` | 4,018 | One day |
| `series_b.csv` | 2,557 | One day |

| Column | Type | Meaning |
| --- | --- | --- |
| `day` | Integer | Consecutive index starting at 1; not a calendar date |
| `flow` | Numeric | Flow value in the original units, m³/s |
| `rain` | Numeric | Daily rainfall value in the original units, mm |

The release replaces calendar dates with day indices and removes original filenames, site labels, and local paths. The rainfall and flow fields retain their original numeric text, row order, and one-day spacing. No values were interpolated, perturbed, rescaled, or removed in preparing the CSV files. Scaling performed during training does not change the files.

The two series have separate anonymous identities. Their indices do not establish a shared calendar. No reverse mapping is distributed. Direct identifiers have been removed; unchanged numerical series should not be treated as a guarantee against linkage to other data.

[Data checksums and row checks](../docs/data_check.json) describe the released files. The loader rejects missing, non-finite, negative, or irregularly spaced records instead of silently joining gaps.

The files are covered by the repository's [Noncommercial Research License](../LICENSE). They are examples for method inspection and reproduction of the supplied runs, not a geographically identified benchmark dataset. Data descriptions in the cited paper refer to that study and should not be assigned to these anonymous files.
