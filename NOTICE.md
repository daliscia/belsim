# Notices

**belsim — Belgium economic and public-finance simulator**
Copyright (C) 2026 Daliscia (https://github.com/daliscia)

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU Affero General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version. It is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See `LICENSE` for the full text.

If you run a modified version on a network server, the licence requires you to offer its
source to the users of that server.

The model is exploratory. It is not an official costing and comes with no guarantee of accuracy;
see `docs/DETAILS.md` ("Validation", "Known limits") and the white paper served at `/whitepaper`.

## Third-party material

The licence above covers the code, the parameter files, the documentation and the texts written
for this project. It does not change the terms of the material below, which is redistributed or
transcribed here with attribution:

| Material | Where | Source and terms |
|---|---|---|
| Statistics downloaded through the Eurostat API | `data/raw/`, `data/base/` | © European Union, Eurostat; reuse authorised with acknowledgement of the source |
| Government accounts, national tax list | `data/raw/revenue/`, `data/base/` | National Bank of Belgium (NBB.Stat), source to be cited on reuse |
| Population movements, fiscal income statistics | `data/raw/demographics/`, `data/raw/revenue/statbel_fisc2024/` | Statbel (Directorate-General Statistics - Statistics Belgium), open data, source to be cited |
| Outlook tables transcribed from the June 2026 economic outlook; measures and costings of the 2024 election programmes (DC2024) | `data/base/revenue_fpb_outlook.csv`, `scenarios/parties/` | Federal Planning Bureau; reproduced with the source cited |
| Official figures used for validation, with their sources and links | `data/reference/validation.yaml` | Study Committee on Ageing, Federal Planning Bureau, National Bank of Belgium, Court of Audit, European Commission and others, as cited per entry |
| Archivo typeface | `web/vendor/archivo-latin-wdth-normal.woff2` | SIL Open Font License 1.1 (`web/vendor/archivo.LICENSE`) |
| js-yaml | `web/vendor/js-yaml.mjs` | MIT License (`web/vendor/js-yaml.LICENSE`) |

Legislated values in `params/` were entered from the official or secondary sources named in each
parameter's `source` field. Check the publishers' own terms before reusing their data outside
this project.

## How this was made

The code, parameter files, documentation and translations were produced in November 2026 by AI
coding agents (Anthropic's Claude, in Claude Code) directed by the author. There has been no
independent expert review.
