# Versions, data locations, and who to ask

Where to get the input data for a given gridding version, and where its output was published.

## Who to ask

| Topic | Contact |
| --- | --- |
| Workflow, harmonisation and downscaling, scenario selection | Jarmo Kikstra |
| Repository structure, QC scripts, extensions pipeline | Annika Högner |
| Cluster runs, papermill drivers, environment | Marco Zecchetto |

If you cannot find an input file and it is not described in [`inputs.md`](inputs.md), ask before
regenerating it — several inputs are expensive to reproduce and some are not reproducible at all
from this repository alone.

## Versions

| Version | Input data (IIASA SharePoint) | Published output | Notes |
| --- | --- | --- | --- |
| `concordia_cmip7_v0-4-0` | [input bundle](https://iiasahub.sharepoint.com/:f:/r/sites/eceprog/Shared%20Documents/Projects/CMIP7/IAM%20Data%20Processing/concordia_cmip7_v0-4-0/input?csf=1&web=1&e=hNAoh5) | see Zenodo record below | Current. Config: `config_cmip7_v0-4-0.yaml` (fast-track), `config_cmip7_v0-4-0-EXT.yaml` (extensions) |
| `concordia_cmip7_esgf_v0_alpha` (v0.3, `0-3-0`) | superseded | — | Config archived under `archive/`; scripts for this version are in `archive/` |
| `concordia_cmip7_v0_2` | superseded | — | Config archived under `archive/` |

The SharePoint links require IIASA credentials and ECE program access.

## Published data and per-version changes

**<https://zenodo.org/records/19730076>**

The Zenodo record is the canonical description of the published datasets and of what changed
between versions. This repository deliberately does not duplicate that changelog — if you need to
know what changed in a release, read the Zenodo record rather than git history.

## Further reading

- input4MIPs dataset overview for ScenarioMIP:
  <https://input4mips-cvs.readthedocs.io/en/latest/dataset-overviews/anthropogenic-slcf-co2-emissions/#scenariomip>
- input4MIPs CVs discussion thread (metadata conventions, reviewer feedback):
  <https://github.com/PCMDI/input4MIPs_CVs/discussions/386>
