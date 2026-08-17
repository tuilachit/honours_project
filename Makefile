UV := uv
PYTHON := $(UV) run python
BASE_CONFIG := configs/base.yaml
WORKFLOW_DIR := configs/workflows

.PHONY: setup data audit-data pilot review-pack run-b0 run-b1 run-b2 run-m1 run-m2 run-m3 eval figures test clean

setup:
	$(UV) sync --frozen

data:
	$(PYTHON) -m scripts.download_data \
		--base-config $(BASE_CONFIG) \
		--condition-config $(WORKFLOW_DIR)/data.yaml

audit-data:
	$(PYTHON) -m scripts.audit_t2_ragbench \
		--base-config configs/audit_base.yaml \
		--dataset-config configs/datasets/t2_ragbench.yaml

pilot:
	$(PYTHON) -m scripts.build_pilot_goldset \
		--base-config $(BASE_CONFIG) \
		--condition-config $(WORKFLOW_DIR)/pilot.yaml

review-pack:
	$(PYTHON) -m scripts.make_source_review_pack \
		--base-config $(BASE_CONFIG) \
		--condition-config $(WORKFLOW_DIR)/pilot.yaml

run-b0:
	$(PYTHON) -m scripts.run_condition \
		--base-config $(BASE_CONFIG) \
		--condition-config configs/b0_flattened_hybrid.yaml

run-b1:
	$(PYTHON) -m scripts.run_condition \
		--base-config $(BASE_CONFIG) \
		--condition-config configs/b1_fact_hybrid.yaml

run-b2:
	$(PYTHON) -m scripts.run_condition \
		--base-config $(BASE_CONFIG) \
		--condition-config configs/b2_structured_lookup.yaml

run-m1:
	$(PYTHON) -m scripts.run_condition \
		--base-config $(BASE_CONFIG) \
		--condition-config configs/m1_candidate_union.yaml

run-m2:
	$(PYTHON) -m scripts.run_condition \
		--base-config $(BASE_CONFIG) \
		--condition-config configs/m2_context_aware.yaml

run-m3:
	$(PYTHON) -m scripts.run_condition \
		--base-config $(BASE_CONFIG) \
		--condition-config configs/m3_hard_negative.yaml

eval: run-b0 run-b1 run-b2 run-m1 run-m2 run-m3

figures:
	$(PYTHON) -m scripts.make_figures \
		--base-config $(BASE_CONFIG) \
		--condition-config $(WORKFLOW_DIR)/figures.yaml

test:
	$(UV) run pytest -q
	$(UV) run mypy src scripts
	$(UV) run ruff check .

clean:
	find results -mindepth 1 ! -name .gitkeep -delete
