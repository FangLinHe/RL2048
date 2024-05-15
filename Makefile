.PHONY: build
.PHONY: tag
# .PHONY: tests
.PHONY: lints
.PHONY: isorts

PKG_VERSION=`hatch version`

# python -m build --sdist --wheel --outdir dist/ .
build:
	hatch build

tag:
	git tag -a v${PKG_VERSION} -m v${PKG_VERSION}
	git push --tag

# tests:
# 	hatch run test:test

lints:
	hatch run test:lint

isorts:
	hatch run test:isort
