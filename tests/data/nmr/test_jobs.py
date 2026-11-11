import gzip
import shutil

from ase.io import read
from jobflow import run_locally

from autoplex.data.common.jobs import check_metallic_castep
from autoplex.data.nmr.jobs import collect_nmr_data

METALLIC_WARNING = """
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !          WARNING! Fractional occupancy of states detected           !
 !                       Magnetic Shielding  is not                    !
 !                     implemented for metallic systems                !
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


def _make_nmr_dir(castep_test_dir, path, metallic):
    """Copy the alpha-cristobalite magres outputs into path (gzipped), optionally adding the metallic warning."""
    source = castep_test_dir / "magres" / "CASTEP_CRISTOBALITE_ALPHA" / "outputs"
    path.mkdir()
    castep_text = (source / "castep.castep").read_text()
    if metallic:
        castep_text += METALLIC_WARNING
    with gzip.open(path / "castep.castep.gz", "wt") as f:
        f.write(castep_text)
    with open(source / "castep.magres", "rb") as f_in, gzip.open(path / "castep.magres.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return path


def test_check_metallic_castep(castep_test_dir, tmp_path):
    normal = _make_nmr_dir(castep_test_dir, tmp_path / "normal", metallic=False)
    metallic = _make_nmr_dir(castep_test_dir, tmp_path / "metallic", metallic=True)

    assert check_metallic_castep(str(normal / "castep.castep.gz")) is False
    assert check_metallic_castep(str(metallic / "castep.castep.gz")) is True


def test_collect_nmr_data_skips_metallic(castep_test_dir, tmp_path, memory_jobstore, clean_dir):
    normal = _make_nmr_dir(castep_test_dir, tmp_path / "normal", metallic=False)
    metallic = _make_nmr_dir(castep_test_dir, tmp_path / "metallic", metallic=True)
    nmr_dirs = [str(normal), str(metallic)]

    # default: the metallic calculation is skipped
    job = collect_nmr_data(nmr_ref_file="nmr_default.extxyz", nmr_dirs=nmr_dirs)
    run_locally(job, create_folders=True, ensure_success=True, store=memory_jobstore)
    structures = read(job.output.resolve(memory_jobstore)["nmr_ref_dir"], index=":")
    assert len(structures) == 1
    assert not structures[0].info["metallic"]

    # use_all_nmr_data=True keeps it, flagged as metallic
    job_all = collect_nmr_data(nmr_ref_file="nmr_all.extxyz", nmr_dirs=nmr_dirs, use_all_nmr_data=True)
    run_locally(job_all, create_folders=True, ensure_success=True, store=memory_jobstore)
    structures_all = read(job_all.output.resolve(memory_jobstore)["nmr_ref_dir"], index=":")
    assert len(structures_all) == 2
    assert sorted(bool(s.info["metallic"]) for s in structures_all) == [False, True]
