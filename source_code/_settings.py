"""Project paths.

All locations are repo-relative by default so the code runs on any machine.
Override any of them with environment variables:

    EIGENSCORE_DATA_DIR   base data dir              (default: <repo>/data)
    EIGENSCORE_MODEL_PATH local model weights dir    (default: <repo>/data/weights)
    EIGENSCORE_OUTPUT_DIR generation output dir      (default: <repo>/data/output)
"""
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

_BASE_DIR = os.environ.get('EIGENSCORE_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))

# Local weight cache. Models given as HuggingFace hub ids (e.g. "meta-llama/Llama-2-7b-hf")
# are resolved through the normal HF cache instead and never touch this path.
MODEL_PATH = os.environ.get('EIGENSCORE_MODEL_PATH', os.path.join(_BASE_DIR, 'weights'))

DATA_FOLDER = os.path.join(_BASE_DIR, 'datasets')
GENERATION_FOLDER = os.environ.get('EIGENSCORE_OUTPUT_DIR', os.path.join(_BASE_DIR, 'output'))

# Sentence-embedding model used for the EigenScore-Output baseline and the
# semantic-similarity correctness measure (paper Section 4.1).
SENTENCE_SIM_MODEL = os.environ.get('EIGENSCORE_SENTSIM_MODEL', 'sentence-transformers/nli-roberta-large')

os.makedirs(GENERATION_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
