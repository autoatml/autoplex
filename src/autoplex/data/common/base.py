from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from atomate2.forcefields.jobs import ForceFieldStaticMaker, ForceFieldRelaxMaker
from jobflow import Flow, Maker
from pydantic import BaseModel, Field
from pymatgen.core import Composition
from pymatgen.core import Structure
from pymatgen.io.ase import MSONAtoms


class DataGenDoc(BaseModel):
    """The inputs used to run this job."""

    database_dir: Path | None = Field(
        None, description="Address to xyz file"
    )
    # or find another way to store the db
    database: list[MSONAtoms] | None = Field(None, description="list of Atoms objects")


@dataclass
class AbstractDataGenFlow(Maker, ABC):
    """
    Base class for data generation workflows.
    """

    static_maker: ForceFieldStaticMaker = None  # labels the data, we might need to use partial here, or we pass mlip configs differently
    isolated_static_maker: ForceFieldStaticMaker = None  # labels isolated atom
    relax_maker: ForceFieldRelaxMaker = None  ## helps with optimization to get data close to minimum (dft or ml model), could be optional
    config: dict = None

    def make(self, input_structures: list[Structure]=None, input_compositions:list[Composition]=None) -> Flow:
        flow = self.data_gen_flow()
        return Flow(flow.jobs,
                    output=DataGenDoc(database_dir=flow.output["database_path"], database=flow.output["database"]))

    @abstractmethod
    def data_gen_flow(self) -> Flow:
        pass
