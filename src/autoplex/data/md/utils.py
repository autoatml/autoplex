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
) -> tuple[np.ndarray, int]:
    """
    Build a per-step temperature schedule for a multi-stage MD run.

    Stage ``i`` holds ``temperature_list[i]`` for ``eqm_step_list[i]`` steps and
    then ramps linearly to the next temperature, spending ``rate_list[i]`` MD steps
    per kelvin. Heating and cooling are treated identically, so the same function
    covers isothermal runs, multi-stage quenches and heating/cooling cycles.

    Parameters
    ----------
    temperature_list : list[float]
        Stage temperatures in K, e.g. [3000, 1500, 300] for a two-stage quench.
    eqm_step_list : list[int] | None
        MD steps to hold each temperature (one entry per temperature).
        Defaults to 10,000 steps per stage.
    rate_list : list[float] | None
        MD steps per kelvin for each ramp (one entry per ramp, i.e.
        len(temperature_list) - 1). 0 jumps straight to the next temperature.
        With a 1 fs time step, rate=10 is 10^14 K/s and rate=100 is 10^13 K/s.
        None, [] or only zeros run without any ramps.

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
    # None, [] or only zeros: no ramps, whatever the length of rate_list
    rates = (
        np.asarray(rate_list, dtype=float)
        if rate_list is not None and np.any(rate_list)
        else np.zeros(max(n_seg - 1, 0))
    )

    if n_seg == 0:
        raise ValueError("temperature_list needs at least one temperature.")
    if holds.size != n_seg:
        raise ValueError(
            f"eqm_step_list needs one entry per temperature ({n_seg}), "
            f"got {holds.size}."
        )
    if (holds < 0).any() or (rates < 0).any():
        raise ValueError("eqm_step_list and rate_list must be non-negative.")

    if rates.size != n_seg - 1:
        raise ValueError(
            f"rate_list needs one rate per ramp between consecutive temperatures: "
            f"{n_seg - 1} for {n_seg} temperature(s), got {rates.size}. "
            "Leave rate_list out, or use only zeros, to run without ramps."
        )

    T_list = []

    for i in range(n_seg):
        # hold current temperature
        T_list.extend([temperature_list[i]] * int(holds[i]))

        if i < n_seg - 1:
            T0, T1 = float(temperature_list[i]), float(temperature_list[i + 1])
            rate_val = float(rates[i])
            if rate_val == 0.0:
                pass
            else:
                md_steps = int(-(T0 - T1) * rate_val)
                tem_interval = max(abs(md_steps), 1)
                # linear interpolation between T0 and T1, one temperature per step
                tem_list = list(np.linspace(T0, T1, tem_interval + 1))[1:]

                T_list.extend(tem_list)

    T_list.append(temperature_list[-1])
    T_array = np.array(T_list, dtype=float)
    n_steps = len(T_array) - 1
    return T_array, n_steps


def handle_md_trajectory(
    traj_path: list | None = None,
    remove_traj_files: bool = False,
) -> tuple[list[list], list[list]]:
    """
    Handle trajectory and associated information.

    Parameters
    ----------
    traj_path: list
        List of paths pointing to trajectory files to be processed.
        Default is None.
    remove_traj_files: bool
        Whether to remove the directories containing trajectory files
        after processing them. Default is False.

    Returns
    -------
    tuple:
        atoms: list
            List of ASE Atoms objects read from the trajectory files.
    """
    atoms = []
    traj_path = [] if traj_path is None else flatten_list(traj_path)

    if all(i is None for i in traj_path):
        raise ValueError("No valid MD trajectory path was obtained!")

    for traj in traj_path:
        if traj is not None and Path(traj).exists():
            logger.info(f"Processing MD trajectory:, {traj}")
            at = ase.io.read(traj, index=":")
            atoms.append(at)

            if remove_traj_files:
                logger.warning(f"The MD trajectory file is deleted: {traj}")
                os.remove(traj)

    return atoms


def plot_temperature_profile(
    T_array: np.ndarray, time_step: float, fig_name: str = "md_temperature_profile.png"
):
    """Plot and save temperature curve."""
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
