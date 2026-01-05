"""Utility functions for MD."""

import logging
import os
from pathlib import Path

import ase.io
import matplotlib.pyplot as plt
import numpy as np

from autoplex.data.common.utils import flatten_list

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_temperature_profile(
    temperature_list: list[float],
    eqm_step_list: list[int] | None = None,
    rate_list: list[float] | None = None,
    time_step: float | None = 1.0,
):
    """
    Generate a temperature array for molecular-dynamics runs.

    Each listed temperature corresponds to one MD step.
    This handles both single-temperature (isothermal) and
    multi-temperature (e.g., multi-stage quench) runs.

    Parameters
    ----------
    temperature_list : list[float]
        A list of values defines a multi-stage quench or anneal profile.
    eqm_step_list : list[int] | None, optional
        Number of MD steps to hold each temperature value.
        Length must match `temperatures`. If None, defaults to 10,000.
    rate_list : list[float] | None, optional
        Relative cooling/heating rates between stages (len = len(temperatures) - 1).
        If None, linear interpolation is used. A larger `rate` value produces more
        intermediate temperatures (slower quench), while a smaller value gives fewer steps
        (faster quench). For example, rate=10 -> 10^14 K/s; rate=100 -> 10^13 K/s.
    dt: float
        Time step for running MD.

    Returns
    -------
    np.ndarray
        Array of temperatures for each MD step (length = n_steps + 1),
        ready to pass to ``ForceFieldMDMaker.temperature``.
    """
    n_seg = len(temperature_list)
    if eqm_step_list is not None and len(eqm_step_list) != n_seg:
        raise ValueError(
            f"Length mismatch: eqm_steps_list ({len(eqm_step_list)}) "
            f"must match temperatures ({n_seg})."
        )

    if n_seg == 1:
        n_hold = 10000 if eqm_step_list is None else eqm_step_list[0]
        T_array = np.array([temperature_list[0]] * (n_hold + 1))
        if time_step is None:
            logging.warning(
                "time_step is not set. Using default value 1 fs for plotting."
            )
            time_step = 1.0
        plot_temperature_profile(T_array, time_step)
        return temperature_list, n_hold

    T_list = []
    eqm_step_list = eqm_step_list or [10000] * n_seg
    if rate_list is None:
        rate_list = [0.0] * (n_seg - 1)

    for i in range(n_seg):
        # hold current temperature
        T_list.extend([temperature_list[i]] * int(eqm_step_list[i]))

        if i < n_seg - 1:
            T0, T1 = float(temperature_list[i]), float(temperature_list[i + 1])
            rate_val = float(rate_list[i])
            if rate_val == 0.0:
                pass
            else:
                md_steps = int(-(T0 - T1) * rate_list[i])
                tem_interval = max(abs(md_steps), 1)
                # linear interpolation between T0 and T1, one temperature per step
                if T0 > T1:
                    tem_list = list(np.linspace(T0, T1, tem_interval + 1))[1:]
                else:
                    tem_list = list(np.linspace(T1, T0, tem_interval + 1))[1:]
                    tem_list.reverse()

                T_list.extend(tem_list)

    T_list.append(temperature_list[-1])
    T_array = np.array(T_list, dtype=float)
    n_steps = len(T_array) - 1
    if time_step is None:
        logging.warning("time_step is not set. Using default value 1 fs for plotting.")
        time_step = 1.0
    plot_temperature_profile(T_array, time_step)
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
            logging.info(f"Processing MD trajectory:, {traj}")
            at = ase.io.read(traj, index=":")
            atoms.append(at)

            if remove_traj_files:
                logging.warning(f"The MD trajectory file is deleted: {traj}")
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

    logging.info("Temperature profile was generated!")
