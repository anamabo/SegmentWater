.PHONY: notebook docs

install_pre-commit:
	pre-commit install

setup_git:
	@echo "Setting up git"
	git init 



## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache