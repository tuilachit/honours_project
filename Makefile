UV := uv
PYTHON := $(UV) run python

.PHONY: setup data run-c1 run-c2a run-c2b run-c3 eval figures test clean

setup:
	$(UV) sync --frozen

data:
	$(PYTHON) scripts/run_condition.py --config configs/base.yaml

run-c1:
	$(PYTHON) scripts/run_condition.py --config configs/c1.yaml

run-c2a:
	$(PYTHON) scripts/run_condition.py --config configs/c2a.yaml

run-c2b:
	$(PYTHON) scripts/run_condition.py --config configs/c2b.yaml

run-c3:
	$(PYTHON) scripts/run_condition.py --config configs/c3.yaml

eval: run-c1 run-c2a run-c2b run-c3

figures:
	$(PYTHON) scripts/make_figures.py --config configs/c3.yaml

test:
	$(UV) run pytest
	$(UV) run mypy src scripts
	$(UV) run ruff check .

clean:
	find results -mindepth 1 ! -name .gitkeep -delete

