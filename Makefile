PYTHON ?= python

.PHONY: smoke matrix-dry-run bash-syntax preflight-full

smoke:
	$(PYTHON) reproduce.py smoke

matrix-dry-run:
	$(PYTHON) reproduce.py matrix --scope all --dry-run

bash-syntax:
	bash -n scripts/run_alfworld_k4_repro.sh

preflight-full:
	$(PYTHON) scripts/full_rollout_preflight.py
