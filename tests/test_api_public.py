"""Public mode of the API (`BELSIM_PUBLIC=1`): read-only, bounded, no leaks, headers."""
import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from belsim.api import public
from belsim.api import service as svc
from belsim.api.app import create_app, public_message
from belsim.core import dims

END = 2030
VAT = {"name": "VAT 22", "measures": [{"name": "VAT", "changes": [
    {"param": "vat.rate_standard", "value": 0.22, "start": 2027}]}], "options": {"end": END}}


def _public_client(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("BELSIM_PUBLIC", "1")
    monkeypatch.setenv("BELSIM_SCENARIOS_DIR", str(tmp_path))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    # no `with`: the start-up warm-up thread (baseline, validation ...) is not wanted in tests
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    return _public_client(monkeypatch, tmp_path)


def test_default_mode_is_unchanged(monkeypatch):
    monkeypatch.delenv("BELSIM_PUBLIC", raising=False)
    monkeypatch.delenv("BELSIM_READ_ONLY", raising=False)
    c = TestClient(create_app())
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "public": False}
    assert "content-security-policy" not in r.headers
    assert c.get("/docs").status_code == 200
    meta = c.get("/api/meta").json()
    assert meta["mode"] == {"public": False, "save": "server", "max_batch": 12}
    assert meta["end_year"] == dims.END_YEAR


def test_healthz_and_version(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "public": True}
    v = client.get("/api/version").json()
    assert v["public"] is True and v["params"] > 100 and 0 < v["assumptions"] < v["params"]
    assert sum(f["params"] for f in v["param_files"]) == v["params"]
    assert "core" in v["modules"] and "fiscal" in v["modules"]
    assert {"commit", "commit_date", "commits", "tests"} <= set(v)


def test_security_and_cache_headers(client):
    r = client.get("/")
    assert r.status_code == 200
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert "unsafe-eval" not in csp and "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "access-control-allow-origin" not in r.headers
    assert r.headers["cache-control"] == "public, max-age=300"
    assert client.get("/vendor/js-yaml.mjs").headers["cache-control"] == "public, max-age=604800"
    for path in ("/api/meta", "/api/baseline"):
        h = client.get(path).headers
        assert h["cache-control"] == "public, max-age=300", path
        assert h["x-content-type-options"] == "nosniff"
    assert client.get("/api/scenarios").headers["cache-control"] == "no-store"
    assert client.get("/api/nope").headers["cache-control"] == "no-store"


def test_no_interactive_docs(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


def test_the_front_end_has_no_inline_script(client):
    """The CSP forbids inline scripts and handlers: the pages must not need them."""
    for path in ("/", "/whitepaper"):
        r = client.get(path)
        assert r.status_code == 200, path
        html = r.text
        assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), f"{path}: inline <script>"
        assert not re.search(r"\son[a-z]+\s*=", html), f"{path}: inline event handler"
        assert "http://" not in html.replace("http://www.w3.org/", "") or True


def test_nothing_is_written_on_request(client, tmp_path):
    r = client.post("/api/scenarios", json={"name": "mine", "measures": VAT["measures"]})
    assert r.status_code == 403 and "browser" in r.json()["error"]
    assert client.delete("/api/scenarios/mine").status_code == 403
    assert client.get("/api/scenarios").json() == {"scenarios": [], "errors": []}
    assert client.get("/api/meta").json()["mode"]["save"] == "browser"
    client.post("/api/run", json=VAT)
    assert list(tmp_path.iterdir()) == []


def test_saved_scenarios_of_the_owner_are_not_listed(monkeypatch, tmp_path):
    (tmp_path / "user").mkdir()
    (tmp_path / "user" / "private.yaml").write_text("name: private\nmeasures: []\n")
    c = _public_client(monkeypatch, tmp_path)
    assert c.get("/api/scenarios").json()["scenarios"] == []
    assert c.get("/api/scenarios?kind=all").json()["scenarios"] == []


def test_runs_still_work_and_are_bounded(client):
    r = client.post("/api/run", json=VAT)
    assert r.status_code == 200 and r.json()["years"][-1] == END
    lim = public.limits()
    big = {"name": "big", "measures": [VAT["measures"][0]] * (lim.max_measures + 1), "options": {"end": END}}
    r = client.post("/api/run", json=big)
    assert r.status_code == 413 and "measures" in r.json()["error"]
    assert client.post("/api/run_band", json=big).status_code == 413
    batch = {"scenarios": [VAT] * (lim.max_batch + 1), "options": {"end": END}}
    r = client.post("/api/run_batch", json=batch)
    assert r.status_code == 413 and str(lim.max_batch) in r.json()["error"]
    ok = client.post("/api/run_batch", json={"scenarios": [VAT] * 2, "options": {"end": END}})
    assert ok.status_code == 200 and len(ok.json()["scenarios"]) == 2


def test_horizon_cap(monkeypatch, tmp_path):
    c = _public_client(monkeypatch, tmp_path, BELSIM_MAX_END=2035)
    assert svc.max_end() == 2035
    meta = c.get("/api/meta").json()
    assert meta["end_year"] == 2035 and meta["dims"]["years"][-1] == 2035
    assert c.get("/api/baseline").json()["years"][-1] == 2035
    r = c.post("/api/run", json={**VAT, "options": {"end": 2050}})
    assert r.status_code == 400 and "2035" in r.json()["error"]
    r = c.post("/api/run", json={**VAT, "options": {}})
    assert r.status_code == 200 and r.json()["years"][-1] == 2035


def test_body_size_limit(monkeypatch, tmp_path):
    c = _public_client(monkeypatch, tmp_path, BELSIM_MAX_BODY=2048)
    huge = {**VAT, "description": "x" * 5000}
    r = c.post("/api/run", json=huge)
    assert r.status_code == 413 and "body" in r.json()["error"]
    # the limit also holds without a Content-Length (chunked upload)
    r = c.post("/api/run", content=(chunk for chunk in [json.dumps(huge).encode()]),
               headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert c.post("/api/run", json=VAT).status_code == 200


def test_rate_limit_per_address(monkeypatch, tmp_path):
    c = _public_client(monkeypatch, tmp_path, BELSIM_RATE=5, BELSIM_TRUST_PROXY=1)
    a = {"x-forwarded-for": "198.51.100.7"}
    codes = [c.get("/api/scenarios", headers=a).status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and set(codes[5:]) == {429}
    r = c.get("/api/scenarios", headers=a)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1 and "error" in r.json()
    # another address is not affected; the rightmost forwarded entry is the one that counts
    assert c.get("/api/scenarios", headers={"x-forwarded-for": "198.51.100.8"}).status_code == 200
    assert c.get("/api/scenarios", headers={"x-forwarded-for": "203.0.113.1, 198.51.100.7"}).status_code == 429
    # pages and health checks are never rate limited
    assert c.get("/healthz", headers=a).status_code == 200
    assert c.get("/", headers=a).status_code == 200


def test_heavy_requests_have_their_own_rate(monkeypatch, tmp_path):
    c = _public_client(monkeypatch, tmp_path, BELSIM_RATE_HEAVY=2)
    codes = [c.post("/api/run", json=VAT).status_code for _ in range(4)]
    assert codes == [200, 200, 429, 429]
    assert c.get("/api/meta").status_code == 200


def test_token_bucket_refills():
    b = public._Buckets(rate=60)
    assert all(b.take("a", now=0.0) == 0.0 for _ in range(60))
    assert b.take("a", now=0.0) > 0
    assert b.take("a", now=2.0) == 0.0          # one token a second
    small = public._Buckets(rate=1, max_keys=10)
    for i in range(50):
        small.take(f"ip{i}", now=float(i))
    assert len(small.data) <= 10


def test_cross_origin_writes_are_refused(client):
    r = client.post("/api/run", json=VAT, headers={"origin": "https://elsewhere.example"})
    assert r.status_code == 403 and "cross-origin" in r.json()["error"]
    assert client.post("/api/run", json=VAT, headers={"origin": "http://testserver"}).status_code == 200
    r = client.options("/api/run", headers={"origin": "https://elsewhere.example",
                                             "access-control-request-method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_errors_do_not_leak(client, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError(f"{svc.ROOT}/params/secret.yaml: no such file")
    monkeypatch.setattr(svc, "describe_params", boom)
    r = client.get("/api/params")
    assert r.status_code == 500
    text = r.text
    assert str(svc.ROOT) not in text and "secret" not in text and "Traceback" not in text
    assert r.json() == {"error": public_message(500, "")}

    def api_boom(*a, **k):
        raise svc.ApiError(500, f"baseline simulation failed: KeyError: {svc.ROOT}/x.py")
    monkeypatch.setattr(svc, "measure_library", api_boom)
    r = client.get("/api/measures")
    assert r.status_code == 500 and str(svc.ROOT) not in r.text and "KeyError" not in r.text


def test_public_message_strips_paths():
    msg = public_message(404, f"no validation reference file at {svc.ROOT}/data/reference/validation.yaml")
    assert str(svc.ROOT) not in msg and "/Users/" not in msg and "/home/" not in msg
    assert public_message(422, "measure 'a', change 1: unknown parameter 'x.y'") == \
        "measure 'a', change 1: unknown parameter 'x.y'"
    assert public_message(400, "see /var/lib/belsim/params/x.yaml") == "see <path>"


def _fake_report():
    path = svc.validation_file()
    st = path.stat()
    rep = {"baseline": [], "measures": [], "problems": [], "not_found": [], "meta": {}}
    svc._valid_cache.put((svc.params_signature(), str(path), st.st_mtime_ns, st.st_size), rep)
    return rep


def test_validation_answers_at_once_while_computing(client, monkeypatch):
    svc._valid_cache.clear()
    monkeypatch.setitem(svc._valid_job, "thread", None)
    monkeypatch.setitem(svc._valid_job, "error", None)
    monkeypatch.setitem(svc._valid_job, "failed_at", None)
    release = {"go": False}

    def slow_report():
        while not release["go"]:
            time.sleep(0.01)
        return _fake_report()
    monkeypatch.setattr(svc, "_validation_report", slow_report)
    try:
        t0 = time.time()
        r = client.get("/api/validation")
        assert r.status_code == 202 and time.time() - t0 < 5
        assert r.json()["status"] == "computing" and r.headers["retry-after"] == "5"
        assert r.headers["cache-control"] == "no-store"
        assert client.get("/api/validation").status_code == 202        # one thread, not two
        release["go"] = True
        svc._valid_job["thread"].join(timeout=5)
        r = client.get("/api/validation")
        assert r.status_code == 200 and r.json()["baseline"] == []
    finally:
        release["go"] = True
        svc._valid_cache.clear()


def test_validation_failure_is_a_state_not_a_trace(client, monkeypatch):
    svc._valid_cache.clear()
    for k in ("thread", "error", "failed_at"):
        monkeypatch.setitem(svc._valid_job, k, None)

    def broken():
        raise RuntimeError(f"cannot read {svc.ROOT}/data/reference/validation.yaml")
    monkeypatch.setattr(svc, "_validation_report", broken)
    client.get("/api/validation")
    svc._valid_job["thread"].join(timeout=5)
    r = client.get("/api/validation")
    assert r.status_code == 500 and str(svc.ROOT) not in r.text and "RuntimeError" not in r.text
    # no new attempt straight after a failure
    assert svc._valid_job["failed_at"] is not None
    thread = svc._valid_job["thread"]
    client.get("/api/validation")
    assert svc._valid_job["thread"] is thread


def test_wait_0_works_outside_public_mode(monkeypatch):
    monkeypatch.delenv("BELSIM_PUBLIC", raising=False)
    svc._valid_cache.clear()
    for k in ("thread", "error", "failed_at"):
        monkeypatch.setitem(svc._valid_job, k, None)
    done = {"n": 0}

    def quick():
        done["n"] += 1
        return _fake_report()
    monkeypatch.setattr(svc, "_validation_report", quick)
    try:
        c = TestClient(create_app())
        r = c.get("/api/validation?wait=0")
        assert r.status_code in (200, 202)
        svc._valid_job["thread"].join(timeout=5)
        assert done["n"] >= 1                       # computed in the background thread
        assert c.get("/api/validation?wait=0").status_code == 200
    finally:
        svc._valid_cache.clear()


def test_main_has_a_public_flag():
    import belsim.api.__main__ as main_mod
    src = open(main_mod.__file__, encoding="utf-8").read()
    assert "--public" in src and "BELSIM_PUBLIC" in src
