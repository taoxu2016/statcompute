# results

Run records and reports from synthetic data. One folder per place and day, for example
`2026-11-14-sandbox/`. Inside it, `runs/<plan>/*.json` are the records the runner wrote, unedited,
and reports rendered from them live in `docs/reports/`.

A record says where it ran. Records from a sandbox or a laptop show that the harness works and how
implementations compare per row on one core. They say nothing about a cluster.
