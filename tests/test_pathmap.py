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
