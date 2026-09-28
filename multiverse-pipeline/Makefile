# Embedding Multiverse Explorer — offline pipeline
# Override the interpreter with `make PY=python ...` if not using the `multiverse` conda env.
PY ?= $(shell conda run -n multiverse which python 2>/dev/null || echo python)
CONFIG ?= config.yaml
export MV_CONFIG = $(CONFIG)

.PHONY: all data run export export-fuzzy export-ec test validate smoke scale clean

all: data run test validate

data:            ## simulate the toy dataset (data/toy.h5ad, ground_truth.csv, gene_sets.json)
	$(PY) data/simulate.py --config $(CONFIG)

run:             ## preprocess -> integrate -> kNN -> metrics -> measures -> regions -> variants -> export
	$(PY) -m pipeline.run_all --config $(CONFIG)

export:          ## re-export web files only
	$(PY) -m pipeline.export

export-fuzzy:    ## add/refresh only the fuzzy UpSet file in the existing web export
	$(PY) -m pipeline.fuzzy_upset --export

export-ec:       ## export the embeddings for the embedding-comparison app (../embedding-comparison/public/data/integration)
	$(PY) -m pipeline.export_ec

test:            ## unit + export-schema + validation tests (documented failures are xfail(strict))
	$(PY) -m pytest -q -rx

validate:        ## print validation checks and write outputs/<name>/validation_results.md
	$(PY) -m pipeline.validate "" composition

smoke:           ## 30-second plumbing check on Iris
	$(PY) -m pipeline.smoke_iris

scale:           ## 50k cells x 48 models scale test (does not touch the web export)
	$(MAKE) data run CONFIG=config.scale_test.yaml

clean:
	rm -rf outputs
