"""Verify .env is ready for Fireworks labeling."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

ok = True
key = os.getenv("FIREWORKS_API_KEY", "")
if not key or key == "your_key_here":
    print("FAIL: set FIREWORKS_API_KEY in .env")
    ok = False
for i in range(4):
    env = f"MODEL_TIER{i}"
    v = os.getenv(env, "")
    if not v or "REPLACE" in v or "..." in v or v.startswith("fake/"):
        print(f"FAIL: set {env} to a real accounts/fireworks/models/... ID")
        ok = False
if ok:
    print("OK: Fireworks env ready for labeling")
sys.exit(0 if ok else 1)
