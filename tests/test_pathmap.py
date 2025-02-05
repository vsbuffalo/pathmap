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

    def test_expand_grid_with_reps(self):
        grid = PathMap({"model": ["A"], "rep": list(range(2))}).expand_grid()
        combinations = grid._state.combinations
        assert len(combinations) == 2
        assert all("rep" in c for c in combinations)
        assert [c["rep"] for c in combinations] == [0, 1]


class TestPathGeneration:
    def test_automatic_directory_structure(self):
        pathset = (
            PathMap({"model": ["A"], "alpha": [0.1]})
            .expand_grid()
            .map_path("output.txt")
        )
        assert len(pathset.paths) == 1
        assert str(pathset.paths[0]) == "model__A/alpha__0.1/output.txt"
        assert pathset.pattern == "model__{model}/alpha__{alpha}/output.txt"

    def test_filename_params_excluded_from_dirs(self):
        pathset = (
            PathMap({"model": ["A"], "alpha": [0.1]})
            .expand_grid()
            .map_path("model_{model}.txt")
        )
        assert len(pathset.paths) == 1
        # 'model' is in filename, so dir is just alpha__0.1
        assert str(pathset.paths[0]) == "alpha__0.1/model_A.txt"
        assert pathset.pattern == "alpha__{alpha}/model_{model}.txt"

    def test_multiple_output_types(self):
        """
        Instead of a single PathSet with a dict of patterns, we create
        separate PathSets for each pattern and test them individually.
        """
        pm = PathMap(
            {"model": ["A"], "alpha": [0.1], "rep": list(range(2))}
        ).expand_grid()

        # Weights paths
        weights = pm.map_path("weights_{rep}.h5")
        assert len(weights.paths) == 2
        assert "weights_0.h5" in str(weights.paths[0])
        assert weights.pattern == "model__{model}/alpha__{alpha}/weights_{rep}.h5"

        # Metrics paths
        metrics = pm.map_path("metrics.csv")
        assert len(metrics.paths) == 2
        assert "metrics.csv" in str(metrics.paths[0])
        assert metrics.pattern == "model__{model}/alpha__{alpha}/rep__{rep}/metrics.csv"


class TestPathSet:
    def test_pathset_single_pattern(self):
        pathset = PathMap({"model": ["A"]}).expand_grid().map_path("model_{model}.h5")
        assert len(pathset.paths) == 1
        assert str(pathset.paths[0]) == "model_A.h5"
        assert pathset.pattern == "model_{model}.h5"

    def test_base_dir(self):
        pathset = (
            PathMap({"model": ["A"]}, base_dir="results")
            .expand_grid()
            .map_path("output.txt")
        )
        assert len(pathset.paths) == 1
        assert str(pathset.paths[0]) == "results/model__A/output.txt"
        assert pathset.pattern == "results/model__{model}/output.txt"


class TestValidationAndErrors:
    def test_invalid_grid_params(self):
        with pytest.raises(ValueError, match="All parameter values must be lists"):
            PathMap({"model": "not_a_list"})

    def test_missing_expand_grid(self):
        grid = PathMap({"model": ["A"]})
        with pytest.raises(ValueError, match="Must generate parameter combinations"):
            grid.map_path("output.txt")


class TestImmutabilityAndViews:
    def test_immutability(self):
        grid1 = PathMap({"model": ["A"]})
        grid2 = grid1.expand_grid()
        assert grid1._state.combinations == []
        assert len(grid2._state.combinations) == 1

    def test_dataframe_view(self):
        grid = PathMap({"model": ["A", "B", "C"]}).expand_grid()
        df = grid.df
        assert len(df) == 3
        assert set(df["model"]) == {"A", "B", "C"}


class TestManifestGeneration:
    def test_basic_manifest(self):
        """Test basic manifest generation with simple grid"""
        pm = PathMap({"model": ["A", "B"], "alpha": [0.1]}).expand_grid()

        # Map multiple path types
        path_sets = pm.map_paths(
            {"weights": "weights.h5", "metrics": "metrics_{model}.csv"}
        )

        # Generate manifest
        manifest = pm.generate_manifest()

        # Check basic properties
        assert len(manifest) == 2  # Two model values
        assert set(manifest.columns) == {
            "model",
            "alpha",
            "weights_path",
            "metrics_path",
        }

        # Verify paths are correct
        first_row = manifest.filter(pl.col("model") == "A").to_dicts()[0]
        # model not in filename, so in directory
        assert first_row["weights_path"].endswith("model__A/alpha__0.1/weights.h5")
        # model in filename, so not in directory
        assert first_row["metrics_path"].endswith("alpha__0.1/metrics_A.csv")

    def test_manifest_with_base_dir(self):
        """Test manifest generation with base directory"""
        pm = PathMap({"model": ["A"], "alpha": [0.1]}, base_dir="results").expand_grid()

        _ = pm.map_paths({"output": "output.txt"})
        manifest = pm.generate_manifest()

        # Check paths include base dir
        assert manifest["output_path"][0].startswith("results/")

    def test_manifest_with_excluded_params(self):
        """Test manifest generation with excluded parameters"""
        pm = PathMap({"model": ["A"], "alpha": [0.1], "rep": [1, 2]}).expand_grid()

        # Exclude rep from directory structure
        _ = pm.map_paths({"metrics": "metrics.csv"}, exclude=["rep"])

        manifest = pm.generate_manifest()

        # Check paths don't include excluded parameter
        assert "rep__" not in manifest["metrics_path"][0]
        # But rep parameter is still in manifest
        assert "rep" in manifest.columns

    def test_manifest_save_load(self, tmp_path):
        """Test saving and loading manifest"""
        output_file = tmp_path / "manifest.csv"

        pm = PathMap({"model": ["A", "B"], "alpha": [0.1]}).expand_grid()
        pm.map_paths({"output": "output_{model}.txt"})

        # Save manifest
        manifest = pm.generate_manifest(str(output_file))

        # Load and verify
        loaded = pl.read_csv(output_file)
        assert len(loaded) == len(manifest)
        assert set(loaded.columns) == set(manifest.columns)

    def test_manifest_without_paths(self):
        """Test manifest generation fails without mapped paths"""
        pm = PathMap({"model": ["A"]}).expand_grid()

        with pytest.raises(
            ValueError, match="Must generate parameter combinations and map paths"
        ):
            pm.generate_manifest()

    def test_manifest_without_expand(self):
        """Test manifest generation fails without expanding grid"""
        pm = PathMap({"model": ["A"]})

        with pytest.raises(ValueError, match="Must generate parameter combinations"):
            # This should fail because we haven't called expand_grid()
            pm.map_paths({"output": "output.txt"})

    def test_manifest_complex_patterns(self):
        """Test manifest with complex path patterns"""
        pm = PathMap({"model": ["A", "B"], "alpha": [0.1], "rep": [1, 2]}).expand_grid()

        _ = pm.map_paths(
            {
                "weights": "model_{model}/rep_{rep}/weights.h5",
                "metrics": "model_{model}/metrics.csv",
                "shared": "shared_file.txt",
            }
        )

        manifest = pm.generate_manifest()

        # Check all expected columns exist
        assert set(manifest.columns) == {
            "model",
            "alpha",
            "rep",
            "weights_path",
            "metrics_path",
            "shared_path",
        }

        # Verify pattern-specific paths
        first_row = manifest.filter(
            (pl.col("model") == "A") & (pl.col("rep") == 1)
        ).to_dicts()[0]

        # Check each path type
        assert first_row["weights_path"].endswith("model_A/rep_1/weights.h5")
        assert first_row["metrics_path"].endswith("model_A/metrics.csv")
        assert first_row["shared_path"].endswith("shared_file.txt")

        # All rows should have same shared path
        assert len(set(manifest["shared_path"].to_list())) == 1
