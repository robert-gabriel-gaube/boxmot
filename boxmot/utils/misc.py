from pathlib import Path
import os

import numpy as np

def ramp_up(x, x_min, x_max, y_min=0.0, y_max=1.0):
    """
    Ascending linear ramp:
      x <= x_min  -> y_min
      x >= x_max  -> y_max
      x_min < x < x_max -> linearly interpolates y_min -> y_max

    Works for scalars or numpy arrays.

    Args:
        x      (float or np.ndarray):      input value(s)
        x_min  (float): lower bound of x
        x_max  (float): upper bound of x
        y_min  (float): output at or below x_min
        y_max  (float): output at or above x_max

    Returns:
        float or np.ndarray: interpolated output
    """
    # avoid zero-division
    if x_max == x_min:
        return np.full_like(x, y_max) if isinstance(x, np.ndarray) else y_max

    if x <= x_min:
        return y_min
    if x >= x_max:
        return y_max
    return y_min + (y_max - y_min) * ((x - x_min) / (x_max - x_min))


def ramp_down(x, x_min, x_max, y_min=0.0, y_max=1.0):
    """
    Descending linear ramp:
      x <= x_min  -> y_max
      x >= x_max  -> y_min
      x_min < x < x_max -> linearly interpolates y_max -> y_min

    Works for scalars or numpy arrays.

    Args:
        x      (float or np.ndarray):      input value(s)
        x_min  (float): lower bound of x
        x_max  (float): upper bound of x
        y_min  (float): output at or above x_max
        y_max  (float): output at or below x_min

    Returns:
        float or np.ndarray: interpolated output
    """
    # avoid zero-division
    if x_max == x_min:
        return np.full_like(x, y_min) if isinstance(x, np.ndarray) else y_min
    
    if x <= x_min:
        return y_max
    if x >= x_max:
        return y_min
    return y_max - (y_max - y_min) * ((x - x_min) / (x_max - x_min))

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it already exists, with an option to create the directory.

    Args:
        path (str or Path): Initial file or directory path.
        exist_ok (bool): If True, returns the original path even if it exists.
        sep (str): Separator to use between path stem and increment.
        mkdir (bool): If True, creates the directory if it doesn’t exist.

    Returns:
        Path: Incremented path, or original if exist_ok is True.
        
    Example:
        runs/exp --> runs/exp2, runs/exp3, etc.
    """
    path = Path(path)  # ensures OS compatibility
    if path.exists() and not exist_ok:
        base, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        
        # Increment path until a non-existing one is found
        for n in range(2, 9999):
            new_path = f"{base}{sep}{n}{suffix}"
            if not Path(new_path).exists():
                path = Path(new_path)
                break

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # creates the directory if it doesn’t exist

    return path