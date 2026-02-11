from ase.build import bulk
from ase.io import read
from jobflow import run_locally, Flow
from autoplex.data.common.flows import DFTStaticLabelling
from pymatgen.io.aims.sets.core import StaticSetGenerator as AimsStaticSetGenerator
from atomate2.aims.jobs.core import StaticMaker as AimsStaticMaker
from autoplex.data.common.jobs import collect_dft_data
from pymatgen.io.ase import AseAtomsAdaptor


def test_dft_labelling_with_aims(memory_jobstore, mock_aims, clean_dir):
    """Test DFTStaticLabelling job with FHi-aims static maker."""
    ref_paths = {
        "static_bulk_0": "bulk_0",
        "static_bulk_1": "bulk_1",
    }

    mock_aims(ref_paths)

    atoms = [bulk("Si", "diamond", a=a) for a in (5.1, 5.2)]
    structures = [AseAtomsAdaptor.get_structure(a) for a in atoms]

    aims_maker = AimsStaticMaker(
        name="static_aims",
        input_set_generator=AimsStaticSetGenerator(
            user_params={
                "species_dir": "light",
                "xc": "pbe",
                "compute_forces": True,
                "compute_analytical_stress": True,
                "override_kgrid_checks": True
            }
        ),
    )

    job_dft = DFTStaticLabelling(
        isolated_atom=False,
        dimer=False,
        static_energy_maker=aims_maker,
    ).make(structures=structures)

    job_collect_data = collect_dft_data(dft_dirs=job_dft.output)

    run_locally(
        Flow([job_dft, job_collect_data]),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    dict_dft = job_collect_data.output.resolve(memory_jobstore)

    path_to_aims = dict_dft['dft_ref_dir']

    atoms = read(path_to_aims, index=":")
    config_types = [at.info['config_type'] for at in atoms]

    assert len(config_types) == 2