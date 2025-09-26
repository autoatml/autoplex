"""Run CASTEP."""


def run_castep(calc):
    """
    Run a CASTEP calculation.

    Parameters
    ----------
    calc : CASTEP calculator
        Running it will generate output files in its
        working directory.

    Returns
    -------
    None
    """
    calc.run()
