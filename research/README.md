# Research workflows

`serra/` is the only research workflow. Start with `serra/README.md`, then
`serra/models.py`. General statistical algorithms live in `bucex/`; these scripts
declare experiments and call that API. Configurations and saved run settings
make every scientific choice explicit. Execution checks are separate from
research-length inference and never serve as publication results.
