"""Streamlit demo for the submission's binary tier0/tier3 router.

Run from any working directory with::

    python -m streamlit run demo/app.py

Routing-only mode is local and free. Fireworks answer generation is optional.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Streamlit places ``demo/`` rather than the repository root on sys.path when
# this file is launched directly. Add the root before importing project modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from agent.fireworks_client import chat_safe
from config import get_model_id_for_tier
from data.schema import CATEGORIES
from demo.logic import BinaryDecision, decide_binary
from router.labels import EASY_CATEGORIES
from router.route_binary import binary_thresholds_from_env


def _configured_for_answers() -> bool:
    required = ("FIREWORKS_API_KEY", "MODEL_TIER0", "MODEL_TIER3")
    return all(os.environ.get(name, "").strip() for name in required)


def _short_model(tier: str) -> str:
    model_id = os.environ.get(f"MODEL_{tier.upper()}", "").strip()
    return model_id.split("/")[-1] if model_id else "(not configured)"


def _render_decision(decision: BinaryDecision, latency_ms: float) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Local routing decision")
        st.success(f"Predicted tier: **{decision.tier}** — 0 routing tokens")
        st.caption(f"{decision.reason} Local latency: {latency_ms:.0f} ms.")
        if decision.probability is None:
            st.info("Category rule bypassed classifier inference.")
        else:
            st.metric("P(cheap_ok)", f"{decision.probability:.3f}")
            chart = pd.DataFrame(
                {
                    "class": ["cheap_ok", "needs_strong"],
                    "probability": [
                        decision.probability,
                        1.0 - decision.probability,
                    ],
                }
            ).set_index("class")
            st.bar_chart(chart)
            st.caption(f"Active threshold: {decision.threshold:.3f}")
    with right:
        st.subheader("Fireworks model mapping")
        st.code(
            os.environ.get(
                f"MODEL_{decision.tier.upper()}",
                f"MODEL_{decision.tier.upper()} is not set",
            ),
            language="text",
        )


def render_app() -> None:
    st.set_page_config(
        page_title="Binary Router Demo", page_icon="↗", layout="wide"
    )

    if "session_tokens" not in st.session_state:
        st.session_state.session_tokens = {"routed": 0, "always_tier3": 0}
    if "log" not in st.session_state:
        st.session_state.log = []

    tau, ner_tau = binary_thresholds_from_env()
    answers_ready = _configured_for_answers()

    with st.sidebar:
        st.header("Binary router")
        st.text(f"tier0: {_short_model('tier0')}")
        st.text(f"tier3: {_short_model('tier3')}")
        st.caption(f"General τ: {tau:.2f} · NER τ: {ner_tau:.2f}")
        st.caption("Cheap-default: " + ", ".join(sorted(EASY_CATEGORIES)))
        st.divider()
        st.metric("Routed answer tokens", st.session_state.session_tokens["routed"])
        st.metric(
            "Measured always-tier3 tokens",
            st.session_state.session_tokens["always_tier3"],
        )
        if st.button("Reset session"):
            st.session_state.session_tokens = {"routed": 0, "always_tier3": 0}
            st.session_state.log = []
            st.rerun()

    st.title("Binary Token-Efficient Router")
    st.caption(
        "AMD Developer Hackathon ACT II — Track 1. Routing runs locally with "
        "zero Fireworks tokens; only optional answers use Fireworks."
    )

    prompt = st.text_area(
        "Query",
        height=120,
        placeholder="e.g. Extract all people and locations from this sentence...",
    )
    selected_category = st.selectbox(
        "Category (optional)", ["(none)"] + CATEGORIES
    )
    category = None if selected_category == "(none)" else selected_category

    generate = st.checkbox(
        "Generate the routed answer via Fireworks",
        value=False,
        disabled=not answers_ready,
        help=(
            "Requires FIREWORKS_API_KEY, MODEL_TIER0, and MODEL_TIER3 in .env."
        ),
    )
    compare = st.checkbox(
        "Compare with always-tier3 (may make one extra Fireworks call)",
        value=False,
        disabled=not generate,
    )

    if not answers_ready:
        st.warning(
            "Routing-only mode: add FIREWORKS_API_KEY, MODEL_TIER0, and "
            "MODEL_TIER3 to .env to enable answers."
        )

    if st.button("Run through router", type="primary", disabled=not prompt.strip()):
        try:
            with st.spinner("Running the local binary router..."):
                started = time.perf_counter()
                decision = decide_binary(prompt, category)
                latency_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 — surface actionable UI errors
            st.error(f"Routing failed: {exc}")
            st.stop()

        _render_decision(decision, latency_ms)

        if generate:
            routed_model = get_model_id_for_tier(decision.tier)
            with st.spinner(f"Generating via {decision.tier}..."):
                routed = chat_safe(
                    routed_model, prompt, max_tokens=700, temperature=0.2
                )

            if routed.get("error"):
                st.error(routed["text"])
                st.stop()

            st.subheader(f"Answer from {decision.tier}")
            st.write(routed["text"])
            routed_tokens = int(routed.get("total_tokens", 0) or 0)
            strongest_tokens: int | None = None
            strongest_answer: dict | None = None

            if compare:
                if decision.tier == "tier3":
                    strongest_answer = routed
                else:
                    with st.spinner("Generating always-tier3 comparison..."):
                        strongest_answer = chat_safe(
                            get_model_id_for_tier("tier3"),
                            prompt,
                            max_tokens=700,
                            temperature=0.2,
                        )
                strongest_tokens = int(
                    strongest_answer.get("total_tokens", 0) or 0
                )
                metric_left, metric_right = st.columns(2)
                metric_left.metric("Routed answer tokens", routed_tokens)
                metric_right.metric(
                    "Always-tier3 tokens",
                    strongest_tokens,
                    delta=strongest_tokens - routed_tokens,
                    delta_color="normal",
                )
                if decision.tier != "tier3" and not strongest_answer.get("error"):
                    with st.expander("Always-tier3 comparison answer"):
                        st.write(strongest_answer["text"])
            else:
                st.metric("Routed answer tokens", routed_tokens)

            st.session_state.session_tokens["routed"] += routed_tokens
            if strongest_tokens is not None:
                st.session_state.session_tokens["always_tier3"] += strongest_tokens
            st.session_state.log.append(
                {
                    "query": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                    "category": category or "unknown",
                    "tier": decision.tier,
                    "p_cheap_ok": decision.probability,
                    "routed_tokens": routed_tokens,
                    "always_tier3_tokens": strongest_tokens,
                }
            )

    if st.session_state.log:
        st.divider()
        st.subheader("Session query log")
        st.dataframe(st.session_state.log, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_app()
