install:
	pip install -e . && pip install -r requirements.txt

notebook:
	jupyter notebook

test:
	pytest tests/

whereami:
	@echo "Current directory: $(shell pwd)"
