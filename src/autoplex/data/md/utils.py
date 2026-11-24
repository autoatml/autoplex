"""Utility functions for MD."""

import logging
import os
from pathlib import Path

import ase.io
import matplotlib.pyplot as plt
import numpy as np

from autoplex.data.common.utils import flatten_list

logger = logging.getLogger(__name__)


def generate_temperature_profile(
    temperature_list: list[float],
    eqm_step_list: list[int] | None = None,
    rate_list: list[float] | None = None,
    ramp_step_list: list[int] | None = None,
) -> tuple[np.ndarray, int]:
    """
    Build a per-step temperature schedule for a multi-stage MD run.

    Stage ``i`` holds ``temperature_list[i]`` for ``eqm_step_list[i]`` steps and
    then ramps linearly to the next temperature. The length of each ramp is either
    given directly (``ramp_step_list``, MD steps per ramp, like in LAMMPS) or follows
    from ``rate_list`` (MD steps per kelvin). Heating and cooling are treated
    identically, so the same function covers isothermal runs, multi-stage quenches
    and heating/cooling cycles.

    Parameters
    ----------
    temperature_list : list[float]
        Stage temperatures in K, e.g. [3000, 1500, 300] for a two-stage quench.
    eqm_step_list : list[int] | None
        MD steps to hold each temperature (one entry per temperature).
        Defaults to 10,000 steps per stage.
    rate_list : list[float] | None
        MD steps per kelvin for each ramp (one entry per ramp, i.e.
        len(temperature_list) - 1); a ramp over dT kelvin takes dT * rate MD steps.
        0 jumps straight to the next temperature.
        With a 1 fs time step, rate=10 is 10^14 K/s and rate=100 is 10^13 K/s.
        None, [] or only zeros run without any ramps.
        Ignored if ramp_step_list is given.
    ramp_step_list : list[int] | None
        MD steps for each ramp (one entry per ramp), like in LAMMPS;
        0 jumps straight to the next temperature. Alternative to rate_list,
        used instead of it if both are given.

    Returns
    -------
    tuple[np.ndarray, int]
        The temperature at every MD step (length n_steps + 1) and n_steps,
        ready for ``ForceFieldMDMaker.temperature`` and ``.n_steps``.
    """
    temps = np.asarray(temperature_list, dtype=float)
    n_seg = temps.size
    holds = (
        np.full(n_seg, 10_000)
        if eqm_step_list is None
        else np.asarray(eqm_step_list, dtype=int)
    )

    if n_seg == 0:
        raise ValueError("temperature_list needs at least one temperature.")
    if holds.size != n_seg:
        raise ValueError(
            f"eqm_step_list needs one entry per temperature ({n_seg}), "
            f"got {holds.size}."
        )
    if (holds < 0).any():
        raise ValueError("eqm_step_list must be non-negative.")

    if ramp_step_list is not None:
        # LAMMPS-style: the number of MD steps per ramp is given directly
        ramp_steps = np.asarray(ramp_step_list, dtype=int)
        if ramp_steps.size != n_seg - 1 or (ramp_steps < 0).any():
            raise ValueError(
                f"ramp_step_list needs one non-negative entry per ramp ({n_seg - 1} "
                f"for {n_seg} temperature(s)), got {list(ramp_step_list)}."
            )
    else:
        # None, [] or only zeros: no ramps, whatever the length of rate_list
        rates = (
            np.asarray(rate_list, dtype=float)
            if rate_list is not None and np.any(rate_list)
            else np.zeros(max(n_seg - 1, 0))
        )
        if rates.size != n_seg - 1:
            raise ValueError(
                f"rate_list needs one rate per ramp between consecutive temperatures: "
                f"{n_seg - 1} for {n_seg} temperature(s), got {rates.size}. "
                "Leave rate_list out, or use only zeros, to run without ramps."
            )
        if (rates < 0).any():
            raise ValueError("rate_list must be non-negative.")
        # dT * (steps per K), at least one step for every rate > 0
        n_rate_steps = (np.abs(np.diff(temps)) * rates).astype(int)
        ramp_steps = np.where(rates > 0, np.maximum(n_rate_steps, 1), 0)

    T_list = []
    for i in range(n_seg):
        # hold the current temperature
        T_list.extend([temps[i]] * int(holds[i]))
        if i < n_seg - 1 and ramp_steps[i] > 0:
            # linear ramp to the next temperature, one temperature per step,
            # without the start (already held) and ending on the next temperature
            T_list.extend(np.linspace(temps[i], temps[i + 1], ramp_steps[i] + 1)[1:])

    T_list.append(temps[-1])
    T_array = np.array(T_list, dtype=float)
    n_steps = len(T_array) - 1
    return T_array, n_steps


def plot_temperature_profile(
    T_array: np.ndarray, time_step: float, fig_name: str = "md_temperature_profile.png"
):
    """Plot and save temperature curve.

    Parameters
    ----------
    T_array : list[float]
        Temperatures in K as array, one entry for each time step.
    time_step : list[float]
        Time step of the simulation.
    fig_name : str
        The name of the file the plot will be saved in. Default: "md_temperature_profile.png"
    """
    time = np.arange(len(T_array)) * float(time_step)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(time / 1000, T_array)
    ax.set_xlabel("Simulation time (ps)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("MD Temperature Profile")
    ax.grid(visible=True, linestyle="--", linewidth=0.5)

    ymax = float(np.max(T_array))
    ymin = float(np.min(T_array))

    ax.axhline(y=ymax, linestyle="--", linewidth=0.8)
    ax.axhline(y=ymin, linestyle="--", linewidth=0.8)

    yticks = list(ax.get_yticks())

    if ymax not in yticks:
        yticks.append(ymax)
    if ymin not in yticks:
        yticks.append(ymin)

    yticks = sorted(yticks)
    ax.set_yticks(yticks)

    out_path = Path(fig_name)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Temperature profile was generated!")
