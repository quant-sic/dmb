venv: $(VIRTUAL_ENV)

tests: venv
	python -X dev -m pytest -m "$(MARK)"

format: venv
	yapf -i --recursive dmb tests scripts
	isort dmb tests scripts

lint: venv
	mypy dmb tests scripts
	pylint -v dmb tests scripts
	yapf --diff $(shell git ls-files | grep '.py$$')
	isort --check-only dmb tests scripts

licenses: venv
	pip-licenses --from=mixed --order=license --summary

.PHONY: help venv tests format lint licenses