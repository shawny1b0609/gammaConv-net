# De-identified demonstration data

These files are supplied to demonstrate the model and its training workflow. They do not represent any particular basin, station, or calendar period.

| File | Rows | Time step |
| --- | ---: | --- |
| `series_a.csv` | 4,018 | One day |
| `series_b.csv` | 2,557 | One day |

| Column | Type | Meaning |
| --- | --- | --- |
| `day` | Integer | Anonymous consecutive index starting at 1 |
| `flow` | Numeric | Flow value in the original units, m³/s |
| `rain` | Numeric | Daily rainfall value in the original units, mm |

The release replaces calendar dates with day indices and removes original filenames, site labels, and local paths. Rainfall and flow retain their original numeric text, row order, and one-day spacing. Training applies scaling in memory.

Each series has an independent anonymous index. The calendar mapping remains private. Numerical sequence matching may still link unchanged values to external records.

[Data checksums and row checks](../docs/data_check.json) describe the released files. The loader rejects missing, non-finite, negative, or irregularly spaced records.

The files support method inspection and reproduction of the supplied runs under the [Noncommercial Research License](../LICENSE). Geographic descriptions in the cited paper apply to its study data; these files carry anonymous identities.
