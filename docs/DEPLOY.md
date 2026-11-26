# Deploying belsim on a public server

`belsim` is one process: a FastAPI application that serves the JSON API under `/api`, the
dashboard at `/` and the methodology page at `/whitepaper`. There is no database, no build step
and no external service; at run time it reads only `params/`, `data/base/`, `data/reference/`,
`scenarios/` and `web/`.

## Run

```
uv sync --no-dev
uv run python -m belsim.api.version --write      # optional: records commit, date and test count
uv run python -m belsim.api --public --host 127.0.0.1 --port 8000
```

`--public` is the same as `BELSIM_PUBLIC=1` (use the variable with `uvicorn belsim.api.app:app`
or in a container; the `Dockerfile` sets it). Without it nothing changes from the local
behaviour: auto-reload, scenarios saved under `scenarios/user/`, interactive API docs.

`belsim/api/build_info.json`, written by the second command, is what `/api/version` and the
methodology page show when the deployed copy has no `.git` directory or no `tests/` (the Docker
image has neither): run it in the checkout before building the image.

Start-up takes a few seconds; the first simulation about two. In public mode a background thread
then computes the baseline, the measure ranking, the party comparison and the validation report
(about a minute of one core). Until the validation report is ready `/api/validation` answers
`202 {"status": "computing"}` and the dashboard shows a waiting state.

## What public mode does

| | |
|---|---|
| Disk | Nothing is written on request. `POST /api/scenarios` and `DELETE /api/scenarios/..` answer 403 and `GET /api/scenarios` is empty, whatever is in `scenarios/user/`. "Save" in the dashboard keeps the scenario in the visitor's browser (`localStorage`); export and import of YAML / JSON files work as before. |
| Size of a request | Body at most 256 KiB (`BELSIM_MAX_BODY`), 80 measures and 400 parameter changes per scenario (`BELSIM_MAX_MEASURES`, `BELSIM_MAX_CHANGES`), 8 scenarios per batch (`BELSIM_MAX_BATCH`), horizon at most `BELSIM_MAX_END` (default 2070, the model's own end; lower it to shorten every run, the dashboard follows). |
| Load | Per address: 240 requests a minute to `/api` (`BELSIM_RATE`), of which 40 that run simulations (`BELSIM_RATE_HEAVY`); answer 429 with `Retry-After`. At most 2 simulation requests at a time (`BELSIM_CONCURRENCY`), 8 waiting (`BELSIM_QUEUE`), then 503. Results are cached in bounded in-memory caches keyed on the parameter files. |
| Errors | A server error answers `{"error": "the server could not compute this; try again later"}`; no message carries a file path, an exception name or a stack trace. Details go to the server log only. |
| Origin | No CORS header is ever sent, so other sites cannot read the API from a browser. `POST` and `DELETE` with an `Origin` other than the site's own host are refused (403). |
| Headers | `Content-Security-Policy` (everything from `'self'`; no inline script, no `eval`; inline styles allowed because the charts use them; images also from `data:`), `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, `X-Frame-Options: DENY`, `Cross-Origin-Opener-Policy`, `Permissions-Policy`. |
| Caching | Pages, scripts and styles `public, max-age=300`; fonts and the vendored library one week; `/api/meta`, `/api/baseline`, `/api/params`, `/api/measures`, `/api/version` five minutes; every other API answer `no-store`. Responses are gzip-compressed. |
| Other | No auto-reload; `/docs`, `/redoc` and `/openapi.json` are not served; `GET /healthz` answers `{"status": "ok"}` without touching the model. |

## Behind a reverse proxy

Run the application on localhost and let the proxy terminate TLS. The application must know the
visitor's address for the rate limit: set `BELSIM_TRUST_PROXY=1` and make sure the proxy
**appends** the peer address to `X-Forwarded-For` (the rightmost entry is the one used, so a
visitor cannot forge it) and forwards `Host`. Never set `BELSIM_TRUST_PROXY` when the
application is reachable directly.

```nginx
server {
    listen 443 ssl http2;
    server_name example.org;                      # your host name
    client_max_body_size 256k;
    add_header Strict-Transport-Security "max-age=31536000" always;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;                  # a batch of party programmes can take a minute
    }
}
```

With Caddy: `reverse_proxy 127.0.0.1:8000` does the same by default. If the public host name
differs from the `Host` the application sees, list it in `BELSIM_ALLOWED_ORIGINS`
(`example.org,www.example.org`), otherwise the dashboard's own POST requests are refused.

One worker is enough for a small audience and keeps one cache and one rate limiter. With
several workers (`uvicorn --workers N`, as in the `Dockerfile`) each has its own caches, its own
limits and its own minute of warm-up: divide the limits accordingly.

## What is not protected

- **A distributed flood.** The limits are per address and per process, in memory. Put the
  site behind a CDN or the proxy's own connection and request limits if that matters.
- **Long requests.** A simulation that has started cannot be interrupted; the bounds above keep
  one request to about a minute at worst (a 40-measure scenario with the per-measure
  breakdown). The proxy's read time-out is the backstop.
- **Authentication.** There is none: everything the server holds is meant to be public. Do
  not put private scenarios in `scenarios/` on a public server (the prebuilt library and the
  party programmes there are served to everyone).
- **TLS, HSTS, logs.** These belong to the proxy.
- **The parameter files.** The server re-reads `params/*.yaml` when they change; anyone who
  can write to that directory changes the model for every visitor.

## Privacy

The application collects no personal data. It has no accounts, sets no cookie, and loads
nothing from a third party: no analytics, no fonts or scripts from a CDN. The browser's
`localStorage` holds the visitor's own preferences and work (theme, language, current scenario,
scenarios kept for comparison, scenarios saved in public mode); it is never sent anywhere except
as the scenario a visitor asks the server to simulate. The server keeps the visitor's address
in memory for the rate limit only, for about a minute, and writes nothing about requests except
what the process or the proxy logs (uvicorn's access log includes the address: turn it off with
`--no-access-log` or let the proxy decide).

## Before publishing

- Supply `web/og.png` (1200 x 630; a placeholder is in the repository) and, if link previews
  matter, make `og:image` and `twitter:image` in `web/index.html` and `web/whitepaper.html`
  absolute URLs: several platforms ignore relative ones.
- Fill in the two placeholders of the methodology page, section "Licence and contact", in
  `web/wp/en.html`, `web/wp/fr.html` and `web/wp/nl.html`.
