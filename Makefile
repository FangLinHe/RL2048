.PHONY: all
.PHONY: build
.PHONY: tests
.PHONY: lints
.PHONY: isorts
.PHONY: tag

PKG_VERSION=`hatch version`

# Equivalent to python -m build --sdist --wheel --outdir dist/ .
all:
	hatch run test:isort
	hatch run test:lint
	hatch run test:test
	hatch build

build:
	hatch build

test:
	hatch run test:test

lint:
	hatch run test:lint

isort:
	hatch run test:isort

tag:
	git tag -a v${PKG_VERSION} -m v${PKG_VERSION}
	git push --tag