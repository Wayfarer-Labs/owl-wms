
.ONESHELL:
ENV_PREFIX = .venv/bin/

.PHONY: help
help:  ## Show available make targets
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: virtualenv
virtualenv:  ## Create a virtual environment with pip
	@echo "Creating virtualenv..."
	@rm -rf .venv
	@python3.12 -m venv .venv
	@$(ENV_PREFIX)pip install --upgrade pip
	@echo "Virtualenv created. Run 'source .venv/bin/activate' to activate it."

.PHONY: install
install:  ## Install project dependencies
	@$(ENV_PREFIX)pip install -r requirements.txt
	@$(ENV_PREFIX)pip install --no-deps git+https://github.com/lucidrains/rotary-embedding-torch.git
	@$(ENV_PREFIX)pip install -r requirements-dev.txt

.PHONY: lint
lint:  ## Run linter
	@$(ENV_PREFIX)black .
	@$(ENV_PREFIX)ruff check .
