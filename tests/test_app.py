"""
AppTest suite for the LLM Audit Suite.

Each test is self-contained: it sets up any required file state, runs the app
headlessly, asserts against rendered output, then cleans up. No API calls are
made — all tests stay within the UI and file-based state layers.
"""

import json
import os
import time
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH     = os.path.join(os.path.dirname(__file__), "..", "app.py")
AUDIT_FILE   = os.path.join(os.path.dirname(__file__), "..", ".audit_state.json")
USAGE_FILE   = os.path.join(os.path.dirname(__file__), "..", ".usage_budget.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def _fresh_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=30)

def _write_audit_state(running: bool, age_seconds: float = 0.0) -> None:
    with open(AUDIT_FILE, "w") as f:
        json.dump({"running": running, "started": time.time() - age_seconds}, f)

def _write_usage(total_tokens: int = 0, audit_starts: list | None = None) -> None:
    with open(USAGE_FILE, "w") as f:
        json.dump({
            "total_tokens": total_tokens,
            "total_calls":  0,
            "by_model":     {},
            "audit_starts": audit_starts or [],
        }, f)

def _cleanup_state_files() -> None:
    for path in (AUDIT_FILE, USAGE_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state():
    """Remove file-based state before and after every test."""
    _cleanup_state_files()
    yield
    _cleanup_state_files()


# ── 1. smoke ──────────────────────────────────────────────────────────────────

def test_app_loads_without_exception():
    """App renders from cold start without raising any unhandled exception."""
    at = _fresh_app().run()
    assert not at.exception, f"Unexpected exception on load: {at.exception}"


# ── 2. navigation ─────────────────────────────────────────────────────────────

def test_tab_order():
    """Tabs must be Storyboard → LLMAuditor → LLM Scorecard, in that order.

    Order matters: st.stop() inside later tabs would blank tabs declared after
    the stop point, so storyboard must be declared first in the source.
    """
    at = _fresh_app().run()
    labels = [t.label for t in at.tabs]
    assert labels[0] == "Storyboard"
    assert labels[1] == "LLMAuditor"
    assert labels[2] == "LLM Scorecard"


# ── 3. audit button state ─────────────────────────────────────────────────────

def test_audit_button_enabled_by_default():
    """Run full audit button is enabled when no audit is in progress."""
    at = _fresh_app().run()
    audit_btn = next((b for b in at.button if "audit" in b.label.lower()), None)
    assert audit_btn is not None, "Audit button not found"
    assert not audit_btn.disabled
    assert audit_btn.label == "Run full audit"


def test_audit_button_disabled_when_globally_locked():
    """Button becomes disabled and relabelled when another session holds the lock.

    This simulates a second browser tab opening while an audit is active —
    the file-based global lock (_is_audit_running_globally) should block it.
    """
    _write_audit_state(running=True, age_seconds=5)
    at = _fresh_app().run()
    audit_btn = next((b for b in at.button if "running" in b.label.lower() or "audit" in b.label.lower()), None)
    assert audit_btn is not None, "Audit button not found"
    assert audit_btn.disabled
    assert audit_btn.label == "Running…"


def test_audit_lock_ignored_after_timeout():
    """A stale lock (older than _AUDIT_TIMEOUT = 120s) does not disable the button.

    Without this, a server crash mid-audit would permanently lock out all users.
    """
    _write_audit_state(running=True, age_seconds=130)
    at = _fresh_app().run()
    audit_btn = next((b for b in at.button if "audit" in b.label.lower()), None)
    assert audit_btn is not None
    assert not audit_btn.disabled
    assert audit_btn.label == "Run full audit"


# ── 4. rate limiting ──────────────────────────────────────────────────────────

def test_rate_limit_cooldown_blocks_rapid_rerun():
    """Clicking Run within 30s of the last audit shows a cooldown error.

    Verifies the 30s cooldown is enforced so users can't spam the Run button.
    """
    _write_usage(audit_starts=[time.time() - 10])  # last run 10s ago
    at = _fresh_app().run()
    audit_btn = next((b for b in at.button if "audit" in b.label.lower()), None)
    assert audit_btn is not None and not audit_btn.disabled
    at = audit_btn.click().run()
    assert at.error, "Expected a cooldown error but none rendered"
    assert "cooldown" in at.error[0].value.lower() or "wait" in at.error[0].value.lower()


def test_rate_limit_hourly_cap_blocks_after_50_runs():
    """Clicking Run after 50 audits in the rolling hour shows a rate-limit error.

    Verifies the _MAX_RUNS_PER_HOUR = 50 cap is enforced.
    """
    recent_starts = [time.time() - i * 60 for i in range(50)]  # 50 runs in the last hour
    _write_usage(audit_starts=recent_starts)
    at = _fresh_app().run()
    audit_btn = next((b for b in at.button if "audit" in b.label.lower()), None)
    assert audit_btn is not None and not audit_btn.disabled
    at = audit_btn.click().run()
    assert at.error, "Expected a rate-limit error but none rendered"
    assert "rate limit" in at.error[0].value.lower() or "max" in at.error[0].value.lower()


# ── 5. token budget ───────────────────────────────────────────────────────────

def test_budget_exhausted_blocks_run():
    """Clicking Run when the token budget is exhausted raises a budget error.

    check_budget() is called before record_audit_start(), so the error fires
    before any API call is made.

    Note: check_budget() is called inside the run path, not at button render time,
    so the button itself stays enabled — the error appears after clicking.
    """
    _write_usage(total_tokens=2_000_001)  # 1 over MAX_TOTAL_TOKENS
    at = _fresh_app().run()
    audit_btn = next((b for b in at.button if "audit" in b.label.lower()), None)
    assert audit_btn is not None and not audit_btn.disabled
    at = audit_btn.click().run()
    assert at.error, "Expected a budget exhaustion error but none rendered"
    # RuntimeError from check_budget() now surfaces its message directly (not swallowed)
    assert "budget" in at.error[0].value.lower() or "token" in at.error[0].value.lower() or "runtimeerror" in at.error[0].value.lower()


# ── 6. FLASK tab ──────────────────────────────────────────────────────────────

def test_flask_tab_shows_prompt_before_run():
    """FLASK tab displays an instructional info message when no run has happened.

    flask_done defaults to False, so the tab should gate on it and show info.
    """
    at = _fresh_app().run()
    assert at.info, "Expected an info message on the FLASK tab before first run"
    assert "run llm scorecard" in at.info[0].value.lower()


def test_flask_tab_unlocks_after_flask_done():
    """FLASK tab renders results (no st.stop gate) when flask_done is True.

    Simulates returning to the tab after a completed FLASK run without
    triggering real API calls.
    """
    at = _fresh_app().run()
    at.session_state["flask_done"] = True
    at = at.run()
    # With flask_done=True the st.stop() gate is skipped — no info message
    flask_info_msgs = [i for i in at.info if "run llm scorecard" in i.value.lower()]
    assert not flask_info_msgs, "FLASK gate info should not show after flask_done=True"
    # Results section renders — app must not have crashed
    assert not at.exception
