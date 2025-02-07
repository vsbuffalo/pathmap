from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import polars as pl


@dataclass
class PathSet:
    """Container for path patterns, concrete paths, and their associated parameters"""

    pattern: str
    paths: List[Path]
    params: List[Dict[str, Any]]


@dataclass(frozen=True)
class GridState:
    """Immutable state container"""

    combinations: List[Dict[str, Any]] = field(default_factory=list)
    path_pattern: Optional[str] = None
    path_patterns: Optional[Dict[str, str]] = None
    base_dir: Optional[Path] = None


class PathMap:
    def __init__(
        self,
        grid_params: Dict[str, List[Any]],
        base_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize a parameter grid for path mapping.

        Args:
            grid_params: Dict mapping parameter names to possible values
            base_dir: Optional base directory for path generation
        """
        self.grid_params = grid_params
        self.base_dir = Path(base_dir) if base_dir else None
        self._state = GridState()
        self._path_sets: Dict[str, PathSet] = {}

        if not all(isinstance(v, list) for v in grid_params.values()):
            raise ValueError("All parameter values must be lists")

    def _with_state(self, **kwargs) -> PathMap:
        """Create new PathMap with updated state"""
        new_map = PathMap(self.grid_params, self.base_dir)
        new_state = GridState(
            combinations=self._state.combinations,
            path_pattern=self._state.path_pattern,
            path_patterns=self._state.path_patterns,
            base_dir=self._state.base_dir,
        )
        for k, v in kwargs.items():
            object.__setattr__(new_state, k, v)
        new_map._state = new_state
        return new_map

    def expand_grid(self) -> PathMap:
        """Generate all parameter combinations"""
        keys = list(self.grid_params.keys())
        values = list(self.grid_params.values())
        product = list(itertools.product(*values))
        combinations = [dict(zip(keys, combo)) for combo in product]
        return self._with_state(combinations=combinations)

    def _extract_filename_params(self, pattern: str) -> Set[str]:
        """Extract parameters used in filename pattern"""
        return set(re.findall(r"\{([^}]+)\}", pattern))

    def _build_directory_path(
        self,
        params: Dict[str, Any],
        filename_params: Set[str],
        exclude: Set,
    ) -> Path:
        """Build directory path from params not used in filename"""
        dir_params = set(params.keys()) - filename_params

        parts = []
        for param in self.grid_params.keys():
            if param in exclude:
                continue
            if param in dir_params:
                value = params[param]
                parts.append(f"{param}__{value}")

        return Path("/".join(parts))

    def _validate_pattern(self, pattern: str) -> None:
        """Validate a path pattern against available parameters"""
        try:
            example = {k: v[0] for k, v in self.grid_params.items()}
            pattern.format(**example)
        except KeyError as err:
            params = list(self.grid_params.keys())
            msg = (
                "Pattern contains undefined parameter "
                f"(pattern: {pattern}, params: {params})."
            )
            raise ValueError(msg) from err
        except ValueError as err:
            msg = f"Invalid pattern syntax: {str(err)}"
            raise ValueError(msg) from err

        if pattern.count("{") != pattern.count("}"):
            msg = (
                f"Mismatched braces in pattern: {pattern.count('{')} "
                f"opening vs {pattern.count('}')} closing"
            )
            raise ValueError(msg)

    def _make_paths(
        self, pattern: str, exclude: Set
    ) -> tuple[List[Path], List[Dict[str, Any]]]:
        """
        Generate concrete paths and their associated parameter combinations
        """
        filename_params = self._extract_filename_params(pattern)
        paths = []
        params = []

        for combo in self._state.combinations:
            dir_path = self._build_directory_path(combo, filename_params, exclude)
            filename = pattern.format(**combo)
            full_path = dir_path / filename
            if self.base_dir:
                full_path = self.base_dir / full_path
            paths.append(full_path)
            params.append(combo.copy())

        return paths, params

    def map_path(self, pattern: str, exclude: List = None) -> PathSet:
        """Map parameter combinations to file paths.

        Args:
            pattern: String pattern with {param} placeholders
            exclude: Parameters to exclude from directory structure
        """
        if not self._state.combinations:
            raise ValueError(
                "Must generate parameter combinations before mapping paths"
            )

        self._validate_pattern(pattern)
        exclude = set() if exclude is None else set(exclude)
        paths, params = self._make_paths(pattern, exclude)
        pattern_path = self._make_pattern(pattern, exclude)

        return PathSet(pattern=pattern_path, paths=paths, params=params)

    def map_paths(
        self, patterns: Dict[str, str], exclude: List = None
    ) -> Dict[str, PathSet]:
        """Map multiple patterns to file paths.

        Args:
            patterns: Dict mapping names to path patterns
            exclude: Parameters to exclude from directory structure
        """
        if not self._state.combinations:
            raise ValueError(
                "Must generate parameter combinations before mapping paths"
            )

        exclude = set() if exclude is None else set(exclude)
        path_sets = {}

        for key, pattern in patterns.items():
            self._validate_pattern(pattern)
            paths, params = self._make_paths(pattern, exclude)
            # For pattern, only use the original exclude list
            pattern_path = self._make_pattern(pattern, exclude)
            path_set = PathSet(pattern=pattern_path, paths=paths, params=params)
            self._path_sets[key] = path_set
            path_sets[key] = path_set

        return path_sets

    def generate_manifest(self, output_path: Optional[str] = None) -> pl.DataFrame:
        """
        Generate a manifest DataFrame combining parameter combinations with paths.

        Args:
            output_path: Optional path to save manifest as CSV
        """
        if not self._state.combinations or not self._path_sets:
            raise ValueError(
                "Must generate parameter combinations and "
                "map paths before creating manifest"
            )

        rows = []
        for combo in self._state.combinations:
            row = combo.copy()

            for key, path_set in self._path_sets.items():
                col_name = f"{key}_path"
                idx = path_set.params.index(combo)
                row[col_name] = str(path_set.paths[idx])

            rows.append(row)

        df = pl.DataFrame(rows)
        if output_path:
            df.write_csv(output_path)
        return df

    def _make_pattern(self, pattern: str, exclude: Set) -> str:
        """Generate wildcard pattern with directory structure"""
        filename_params = self._extract_filename_params(pattern)
        dir_parts = []

        for param in self.grid_params.keys():
            if param in exclude:
                continue
            if param not in filename_params:
                dir_parts.append(f"{param}__{{{param}}}")

        dir_pattern = "/".join(dir_parts)

        if dir_pattern:
            full_pattern = f"{dir_pattern}/{pattern}"
        else:
            full_pattern = pattern

        if self.base_dir:
            full_pattern = f"{self.base_dir}/{full_pattern}"

        return full_pattern.lstrip("/")

    @property
    def df(self) -> Optional[pl.DataFrame]:
        """View current parameter combinations as DataFrame"""
        if not self._state.combinations:
            return None
        return pl.DataFrame(self._state.combinations)
