.PHONY: all
.PHONY: build
.PHONY: install
.PHONY: clean
.PHONY: test
.PHONY: lint
.PHONY: isort
.PHONY: mypy
.PHONY: tag

PKG_VERSION=`hatch version`

# Equivalent to python -m build --sdist --wheel --outdir dist/ .
all:
	hatch run test:isort
	hatch run test:lint
	mypy rl_2048 tests --check-untyped-defs
	hatch run test:test
	hatch build

build:
	hatch build

install:
	pip install .

clean:
	rm -rf dist
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

test:
	hatch run test:test

lint:
	hatch run test:lint

isort:
	hatch run test:isort

mypy:
	mypy rl_2048 tests --check-untyped-defs

tag:
	git tag -a v${PKG_VERSION} -m v${PKG_VERSION}
	git push --tag