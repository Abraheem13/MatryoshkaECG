.PHONY: help smoke test train all clean paper

help:
	@echo "make smoke   validate the whole pipeline on synthetic data (~2 min)"
	@echo "make test    run unit tests"
	@echo "make all     full experiment matrix (see run_all.sh for knobs)"
	@echo "make paper   aggregate results and rebuild the manuscript"
	@echo "make clean   remove results (keeps data/ and checkpoints/)"

smoke:
	python scripts/smoke_test.py --full

test:
	python tests/test_stats.py

all:
	./run_all.sh

paper:
	python scripts/aggregate_results.py --results-dir results --paper-dir ../paper
	python scripts/make_figures.py --results-dir results --out-dir ../paper/figures
	$(MAKE) -C ../paper check

clean:
	rm -rf results/runs results/logs results/*.json
