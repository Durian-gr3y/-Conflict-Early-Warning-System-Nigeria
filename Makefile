.PHONY: env install ingest phase1 test reproduce clean

# ─── Environment ──────────────────────────────────────────────────────────────
env:
	python -m venv .venv
	@echo "Run: .venv\\Scripts\\activate  (Windows)"

install:
	pip install -r requirements.txt

# ─── Data ingestion (Phase 2+) ────────────────────────────────────────────────
ingest:
	python src/ingestion/run_all.py

# ─── Notebooks ────────────────────────────────────────────────────────────────
phase1:
	jupyter nbconvert --to notebook --execute \
		notebooks/01_eda_baseline.ipynb \
		--output notebooks/01_eda_baseline_executed.ipynb \
		--ExecutePreprocessor.timeout=600

phase2:
	python notebooks/run_phase2.py

phase2-download:
	python notebooks/run_phase2.py --download

phase3:
	python notebooks/run_phase3.py

phase4:
	python notebooks/run_phase4.py

phase5:
	python notebooks/run_phase5.py

phase6:
	python -m unittest tests/test_scaling.py

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	python -m unittest tests/test_models.py tests/test_delivery.py tests/test_scaling.py

# ─── Reproduce everything from scratch ────────────────────────────────────────
reproduce:
	python notebooks/run_reproduce_all.py

# ─── Dashboard (Phase 5) ──────────────────────────────────────────────────────
dashboard:
	streamlit run app/dashboard.py

# ─── Clean generated outputs ─────────────────────────────────────────────────
clean:
	find . -name "*_executed.ipynb" -delete
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
