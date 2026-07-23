CONFIG ?= configs/synthetic.yaml
PYTHON ?= python

.PHONY: all p10

# One resumable command validates or builds P1-P10 in order.
all: p10

p10:
	$(PYTHON) scripts/run_full_pipeline.py --config $(CONFIG)
