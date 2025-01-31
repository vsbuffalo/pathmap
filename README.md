# PathMap

[![Python CI](https://github.com/vsbuffalo/pathmap/actions/workflows/ci.yml/badge.svg)](https://github.com/vsbuffalo/pathmap/actions/workflows/ci.yml)

PathMap is a Python library that helps you organize files based on parameter
combinations by creating structured directory paths. It's particularly useful
for managing experiment outputs, multiple models' output, and data processing
pipelines where you need to track multiple parameter configurations.

## Installation

```bash
pip install git+https://github.com/vsbuffalo/pathmap.git
```

You can also install in development mode:

```bash
git clone https://github.com/vsbuffalo/pathmap.git
cd pathmap
pip install -e .
```

## Quick Start

```python
from pathmap import PathMap

# Define your parameter grid
params = {
    "model": ["A", "B"],
    "alpha": [0.001, 0.01]
}

# Generate a cross-product grid of all
# parameters, and develop a path mapping
# based on the provided filename path patterns.
paths = (PathMap(params)
         .expand_grid()
         .map_paths({
             "fits": "model_fit.tsv",
             "metrics": "metrics.tsv"
         }))

# Access generated paths
for path in paths["fits"].paths:
    print(path)
# Output:
# model__A/alpha__0.001/model_fit.tsv
# model__A/alpha__0.01/model_fit.tsv
# model__B/alpha__0.001/model_fit.tsv
# model__B/alpha__0.01/model_fit.tsv
```

## Example Usage

Creating a grid with replicates:

```python
paths = (PathMap(params)
         .expand_grid(nreps=3)  # Create 3 replicates
         .map_path("output_{rep}.txt"))
```

Note that you can use `PathMap(rep_name = "seed")` if you want replicates to be
used as seeds (if not, it will use the parameter name "rep").

One can use a `base_dir` prefix:

```python
paths = (PathMap(params, base_dir="results")
         .expand_grid()
         .map_path("output.txt"))
```

The `PathMap.map_paths()` method takes in a dictionary of filename patterns to
instantiate with the grid parameters.

```python
paths = (PathMap(params)
         .expand_grid()
         .map_paths({
             "model": "model_{model}.tsv",
             "config": "config.yaml",
             "metrics": "metrics_{model}.tsv"
         }))
```

A Polars dataframe of all parameters, paths, etc can be accessed with:

```python
grid = PathMap(params).expand_grid()
df = grid.df
```

## Path Pattern Rules

- Parameters in filenames are excluded from directory structure
- Directory names are formatted as `parameter__value`
- Patterns support Python string formatting syntax
- Optional base directory prepends all paths

## Development

To run tests:

```bash
pytest tests/
```
