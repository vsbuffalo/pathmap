from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import polars as pl


@dataclass
class PathSet:
    """Container for both path patterns and concrete paths"""

    pattern: str
    paths: List[Path]

    def format_pattern(self, func: callable) -> str:
        """
        Transform parameters in pattern string using provided function.
        """
        if not self.pattern:
            raise ValueError("No pattern set")

        # Find all parameter names between curly braces
        import re

        params = re.findall(r"\{([^}]+)\}", self.pattern)

        # Replace each parameter with transformed version
        result = self.pattern
        for param in params:
            result = result.replace(f"{{{param}}}", f"{{{func(param)}}}")

        return result


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
        """
        Initialize parameter grid.

        Args:
            grid_params: Dict mapping parameter names to possible values
            base_dir: Optional base directory for path generation
        """
        self.grid_params = grid_params
        self.base_dir = Path(base_dir) if base_dir else None
        self._state = GridState()

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
        # Generate base combinations
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
        # Parameters that should be directories (not in filename)
        dir_params = set(params.keys()) - filename_params

        # Use original parameter order from grid_params
        parts = []
        for param in self.grid_params.keys():  # Use original order
            if param in exclude:
                continue
            if param in dir_params:  # Only include if it's a directory parameter
                value = params[param]
                parts.append(f"{param}__{value}")

        return Path("/".join(parts))

    def _validate_pattern(self, pattern: str) -> None:
        """
        Validate a path pattern against available parameters.
        """
        try:
            # Create example with both grid params and any special params (like rep)
            example = {k: v[0] for k, v in self.grid_params.items()}
            pattern.format(**example)
        except KeyError as err:
            msg = f"Pattern contains undefined parameter (pattern: {pattern}, params: {list(self.grid_params.keys())})."
            raise ValueError(msg) from err
        except ValueError as err:
            msg = f"Invalid pattern syntax: {str(err)}"
            raise ValueError(msg) from err

        open_count = pattern.count("{")
        close_count = pattern.count("}")
        if open_count != close_count:
            msg = (
                f"Mismatched braces in pattern: "
                f"{open_count} opening vs {close_count} closing"
            )
            raise ValueError(msg)

    def map_path(self, pattern: str, exclude: List = None) -> PathSet:
        """Map parameter combinations to file paths, returning the PathSet."""
        if not self._state.combinations:
            raise ValueError(
                "Must generate parameter combinations before mapping paths"
            )

        # Validate pattern
        self._validate_pattern(pattern)

        exclude = set() if exclude is None else set(exclude)
        concrete_paths = self._make_paths(pattern, exclude)
        pattern_path = self._make_pattern(pattern, exclude)

        return PathSet(pattern=pattern_path, paths=concrete_paths)

    def map_paths(
        self, patterns: Dict[str, str], exclude: List = None
    ) -> Dict[str, PathSet]:
        """Map a dictionary of patterns to their own PathSet"""
        assert isinstance(patterns, dict), f"patterns must be a dict, got: {patterns}"
        return {k: self.map_path(p, exclude) for k, p in patterns.items()}

    def _make_paths(self, pattern: str, exclude: Set) -> List[Path]:
        """Internal method to generate concrete paths"""
        filename_params = self._extract_filename_params(pattern)
        paths = []

        for combo in self._state.combinations:
            dir_path = self._build_directory_path(combo, filename_params, exclude)
            filename = pattern.format(**combo)
            full_path = dir_path / filename
            if self.base_dir:
                full_path = self.base_dir / full_path
            paths.append(full_path)

        return paths

    def _make_pattern(self, pattern: str, exclude: Set) -> str:
        """Internal method to generate wildcard patterns"""
        filename_params = self._extract_filename_params(pattern)
        dir_parts = []

        # Only create directory structure for parameters not used in filename
        for param in self.grid_params.keys():
            if param in exclude:
                continue
            if param not in filename_params:
                dir_parts.append(f"{param}__{{{param}}}")

        dir_pattern = "/".join(dir_parts)

        # Replace params in filename with wildcards
        # for param in filename_params:
        # pattern = pattern.replace(f"{{{param}}}", f"{{wildcards.{param}}}")

        if dir_pattern:
            full_pattern = f"{dir_pattern}/{pattern}"
        else:
            full_pattern = pattern

        if self.base_dir:
            full_pattern = f"{self.base_dir}/{full_pattern}"

        return full_pattern.lstrip("/")

    @property
    def df(self) -> Optional[pl.DataFrame]:
        """View combinations as DataFrame when needed"""
        if not self._state.combinations:
            return None
        return pl.DataFrame(self._state.combinations)
