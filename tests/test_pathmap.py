import polars as pl
import pytest

from pathmap import PathMap


@pytest.fixture
def basic_grid():
    return PathMap({"model": ["A", "B", "C"], "alpha": [0.1, 0.2]})


@pytest.fixture
def expanded_grid():
    return PathMap({"model": ["A", "B", "C"], "alpha": [0.1, 0.2]}).expand_grid()


class TestPathMapBasics:
    def test_basic_grid_creation(self, basic_grid):
        assert basic_grid.grid_params["model"] == ["A", "B", "C"]
        assert basic_grid.grid_params["alpha"] == [0.1, 0.2]

    def test_expand_grid(self, expanded_grid):
        combinations = expanded_grid._state.combinations
        assert len(combinations) == 6
        assert {"model": "A", "alpha": 0.1} in combinations
        assert {"model": "C", "alpha": 0.2} in combinations


class TestPathGeneration:
    def test_directory_structure_with_params(self):
        """Test directory structure is built correctly with params"""
        pathset = (
            PathMap({"model": ["A"], "alpha": [0.1]})
            .expand_grid()
            .map_path("output.txt")
        )
        assert len(pathset.paths) == 1
        assert len(pathset.params) == 1
        assert str(pathset.paths[0]) == "model__A/alpha__0.1/output.txt"
        assert pathset.params[0] == {"model": "A", "alpha": 0.1}

    def test_filename_params_with_directory(self):
        """Test params used in filename are excluded from directory"""
        pathset = (
            PathMap({"model": ["A"], "alpha": [0.1]})
            .expand_grid()
            .map_path("model_{model}.txt")
        )
        assert len(pathset.paths) == 1
        assert len(pathset.params) == 1
        assert str(pathset.paths[0]) == "alpha__0.1/model_A.txt"
        assert pathset.params[0] == {"model": "A", "alpha": 0.1}

    def test_multiple_paths_with_params(self):
        """Test multiple paths with different parameter usages"""
        pm = PathMap({"model": ["A", "B"], "rep": [1, 2]}).expand_grid()

        pathsets = pm.map_paths(
            {
                "weights": "weights_{rep}.h5",  # Uses rep in filename
                "metrics": "metrics_{model}.csv",  # Uses model in filename
                "shared": "shared.txt",  # Uses no params in filename
            }
        )

        # Check weights paths and params
        weights = pathsets["weights"]
        assert len(weights.paths) == 4
        assert len(weights.params) == 4
        for path, param in zip(weights.paths, weights.params):
            # Check value appears in path
            assert str(param["rep"]) in str(path)
            assert f"model__{param['model']}" in str(path)  # Check directory structure

        # Check metrics paths and params
        metrics = pathsets["metrics"]
        assert len(metrics.paths) == 4
        assert len(metrics.params) == 4
        for path, param in zip(metrics.paths, metrics.params):
            assert str(param["model"]) in str(path)
            assert "rep__" in str(path)  # Check rep is in directory structure

        # Check shared paths and params
        shared = pathsets["shared"]
        assert len(shared.paths) == 4
        assert len(shared.params) == 4
        assert all("shared.txt" in str(p) for p in shared.paths)
        # Check both params appear in directory structure
        for path, param in zip(shared.paths, shared.params):
            assert f"model__{param['model']}" in str(path)
            assert f"rep__{param['rep']}" in str(path)


class TestManifestGeneration:
    def test_manifest_path_param_correspondence(self):
        """Test manifest correctly matches paths with parameters"""
        pm = PathMap({"model": ["A", "B"], "alpha": [0.1, 0.2]}).expand_grid()

        pm.map_paths({"output": "model_{model}/alpha_{alpha}.txt"})

        manifest = pm.generate_manifest()

        # Check all combinations are present
        assert len(manifest) == 4

        # Verify specific path-parameter matches
        for model in ["A", "B"]:
            for alpha in [0.1, 0.2]:
                rows = manifest.filter(
                    (pl.col("model") == model) & (pl.col("alpha") == alpha)
                ).to_dicts()
                assert len(rows) == 1
                path = rows[0]["output_path"]
                assert f"model_{model}/alpha_{alpha}.txt" in path

    def test_manifest_with_excluded_params(self):
        """Test manifest generation with excluded parameters"""
        pm = PathMap({"model": ["A"], "rep": [1, 2]}).expand_grid()

        pm.map_paths({"output": "output_{model}.txt"}, exclude=["rep"])

        manifest = pm.generate_manifest()

        # Check rep stays in manifest but not in paths
        assert "rep" in manifest.columns
        assert all("rep__" not in p for p in manifest["output_path"])

        # Check paths are correct for each combination
        paths = manifest["output_path"].to_list()
        assert len(set(paths)) == 1  # Same path for both reps

    def test_manifest_complex_path_patterns(self):
        """Test manifest with complex nested paths"""
        pm = PathMap(
            {"model": ["A", "B"], "dataset": ["train", "test"], "rep": [1, 2]}
        ).expand_grid()

        pm.map_paths({"nested": "model_{model}/data_{dataset}/rep_{rep}.txt"})

        manifest = pm.generate_manifest()

        # Check all combinations exist
        assert len(manifest) == 8

        # Verify path structure for each combination
        for model in ["A", "B"]:
            for dataset in ["train", "test"]:
                for rep in [1, 2]:
                    rows = manifest.filter(
                        (pl.col("model") == model)
                        & (pl.col("dataset") == dataset)
                        & (pl.col("rep") == rep)
                    ).to_dicts()
                    assert len(rows) == 1
                    path = rows[0]["nested_path"]
                    assert f"model_{model}/data_{dataset}/rep_{rep}.txt" in path
