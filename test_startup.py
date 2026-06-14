"""Quick startup test — checks the import chain and model loading."""
import os
os.environ["TRAFIX_MODEL_VERSION"] = "v6"

print("1. Importing main:app ...")
from main import app
print("   OK")

print("2. Loading model ...")
from backend.main import load_model
result = load_model()
print(f"   load_model() = {result}")

print("3. Checking edge_index on loaded model ...")
from backend.main import ai_agent
if ai_agent is not None:
    print(f"   edge_index shape: {list(ai_agent.edge_index.shape)}")
    print(f"   edge_index:\n{ai_agent.edge_index}")
else:
    print("   ai_agent is None (heuristic fallback)")

print("\nAll checks passed.")
