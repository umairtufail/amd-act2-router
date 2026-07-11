"""Streamlit demo: watch the multi-tier router decide, live.

Run:  pip install -r requirements-demo.txt && streamlit run demo/app.py

The router decision is always shown (local, free). Actually generating the
answer via Fireworks is optional and requires env vars — the demo works for
routing-only walkthroughs without any API key.
"""

import os

import streamlit as st

from data.schema import CATEGORIES

st.set_page_config(page_title="Multi-Tier Router Demo", page_icon="=", layout="wide")
st.title("Multi-Tier Token-Efficient Router")
st.caption("AMD Developer Hackathon ACT II — Track 1. Routing is local (zero "
           "Fireworks tokens); only answers cost tokens.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Wired-up tiers")
    creds_ok = bool(os.environ.get("FIREWORKS_API_KEY"))
    for tier_env in ("MODEL_TIER0", "MODEL_TIER1", "MODEL_TIER2", "MODEL_TIER3"):
        st.text(f"{tier_env}: {os.environ.get(tier_env, '(not set)')}")
    st.divider()
    if "session_tokens" not in st.session_state:
        st.session_state.session_tokens = {"router": 0, "always_strongest": 0}
        st.session_state.log = []
    st.metric("Session tokens (router)", st.session_state.session_tokens["router"])
    st.metric("Session tokens (always-strongest)",
              st.session_state.session_tokens["always_strongest"])

# ---------------------------------------------------------------- inputs
prompt = st.text_area("Query", height=120,
                      placeholder="e.g. A tank starts with 480 liters...")
category = st.selectbox("Category (optional)", ["(none)"] + CATEGORIES)
category = None if category == "(none)" else category
generate = st.checkbox("Also generate the answer via Fireworks (costs tokens)",
                       value=False, disabled=not creds_ok,
                       help="Requires FIREWORKS_API_KEY and MODEL_TIER0..3 env vars.")

if st.button("Run through router", type="primary") and prompt.strip():
    from router.infer_multitier_router import checkpoint_available, predict_tier_proba

    if not checkpoint_available():
        st.error("No trained checkpoint found. Run: python -m router.train_multitier_router")
        st.stop()

    probs = predict_tier_proba(prompt, category)
    tier = max(probs, key=probs.get)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Local routing decision")
        st.success(f"Predicted tier: **{tier}** — 0 routing tokens")
        st.bar_chart(probs)
    with col2:
        st.subheader("Model mapping")
        model_env = f"MODEL_{tier.upper()}"
        st.code(os.environ.get(model_env, f"${model_env} not set"), language="text")

    if generate:
        from agent.fireworks_client import chat_safe
        from config import get_model_id_for_tier, get_tier_names

        strongest = get_tier_names()[-1]
        with st.spinner(f"Generating with {tier} and (for comparison) {strongest}..."):
            routed = chat_safe(get_model_id_for_tier(tier), prompt)
            strongest_ans = (routed if tier == strongest else
                             chat_safe(get_model_id_for_tier(strongest), prompt))

        st.subheader(f"Answer from {tier}")
        st.write(routed["text"])
        c1, c2 = st.columns(2)
        c1.metric(f"Tokens via router ({tier})", routed["total_tokens"])
        c2.metric(f"Tokens via always-{strongest}", strongest_ans["total_tokens"],
                  delta=routed["total_tokens"] - strongest_ans["total_tokens"],
                  delta_color="inverse")

        st.session_state.session_tokens["router"] += routed["total_tokens"]
        st.session_state.session_tokens["always_strongest"] += strongest_ans["total_tokens"]
        st.session_state.log.append({
            "query": prompt[:60] + ("..." if len(prompt) > 60 else ""),
            "tier": tier,
            "router_tokens": routed["total_tokens"],
            "strongest_tokens": strongest_ans["total_tokens"],
        })

if st.session_state.get("log"):
    st.divider()
    st.subheader("Session query log")
    st.dataframe(st.session_state.log, use_container_width=True)
