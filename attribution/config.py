from pathlib import Path

# --- Paths to amao's results ---
AMAO_RESULTS_DIR = Path("/projectnb/cs505am/students/amao/results")
IN_CTX_RESULTS   = AMAO_RESULTS_DIR / "olmo2-1b-instruct-in-cxt-icl5-results.json"
RECALL_RESULTS   = AMAO_RESULTS_DIR / "olmo2-1b-instruct-recall-icl5-results.json"

# --- Model ---
MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"

# --- Output directory for attribution results ---
ATTR_RESULTS_DIR = Path(__file__).parent / "results"
ATTR_RESULTS_DIR.mkdir(exist_ok=True)
