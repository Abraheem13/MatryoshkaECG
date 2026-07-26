PAPER = main
RESP  = response_to_reviewers

.PHONY: all paper response clean check

all: paper response check

paper:
	latexmk -pdf -interaction=nonstopmode $(PAPER).tex

response:
	latexmk -pdf -interaction=nonstopmode $(RESP).tex

# hard gate: no placeholder may survive to submission
check: paper
	@n=$$(pdftotext $(PAPER).pdf - | grep -c "\[TBD\]" || true); \
	if [ "$$n" -gt 0 ]; then \
	  echo "FAIL: $$n placeholder [TBD] markers remain in $(PAPER).pdf."; \
	  echo "      Run: python ../code/scripts/aggregate_results.py"; \
	  exit 1; \
	else echo "OK: no placeholders remain."; fi

clean:
	latexmk -C
	rm -f *.bbl *.aux *.log *.out *.blg
