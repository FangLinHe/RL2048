.PHONY: build
.PHONY: tag
.PHONY: tests
.PHONY: lints
.PHONY: isorts

PKG_VERSION=`hatch version`

# Equivalent to python -m build --sdist --wheel --outdir dist/ .
build:
	hatch build

tag:
	git tag -a v${PKG_VERSION} -m v${PKG_VERSION}
	git push --tag

test:
	hatch run test:test

lint:
	hatch run test:lint

isort:
	hatch run test:isort
