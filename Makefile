.PHONY: test run verify

PYTHON := uv run python

test:
	uv run pytest -q

run:
	$(PYTHON) -m src.main $(AUDIO)

verify:
	$(PYTHON) -m src.preprocess.verify $(AUDIO) $(if $(OUTPUT),--output $(OUTPUT),)
