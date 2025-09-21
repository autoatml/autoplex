from autoplex.data.common.flows import DFTStaticLabelling
from autoplex.castep.jobs import BaseCastepMaker
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import os
os.environ["castep_command"]="/usr/local/CASTEP-20/castep.mpi"

class TestDFTStaticLabellingConfiguration:
    """Unit tests for DFTStaticLabelling configuration with real structures and makers."""
    
    def test_dft_labelling_with_castep_maker(self):
        """Test DFTStaticLabelling configuration with real BaseCastepMaker."""
        # Create test structure
        si_ase = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        si_pmg = AseAtomsAdaptor.get_structure(si_ase)
        
        # Create real CASTEP maker
        castep_maker = BaseCastepMaker(
            name="dft_test",
            cut_off_energy=200.0,
            kspacing=0.5,
            xc_functional="PBE",
            task="SinglePoint",
        )
        
        # Create DFTStaticLabelling with real maker
        dft_labelling = DFTStaticLabelling(
            static_energy_maker=castep_maker,
            isolated_atom=True,
            isolatedatom_box=[20.0, 20.5, 21.0],
        )
        
        # Test configuration
        assert dft_labelling.static_energy_maker == castep_maker
        assert dft_labelling.isolated_atom is True
        assert dft_labelling.isolatedatom_box == [20.0, 20.5, 21.0]
        assert isinstance(dft_labelling.static_energy_maker, BaseCastepMaker)
    
    def test_dft_labelling_with_dimer_configuration(self):
        """Test DFTStaticLabelling dimer configuration with real structures."""
        # Create multiple test structures
        si_ase = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        ge_ase = Atoms("Ge", positions=[[0, 0, 0]], cell=[12, 12, 12], pbc=True)
        
        si_pmg = AseAtomsAdaptor.get_structure(si_ase)
        ge_pmg = AseAtomsAdaptor.get_structure(ge_ase)
        test_structures = [si_pmg, ge_pmg]
        
        # Create CASTEP maker
        castep_maker = BaseCastepMaker(
            name="dimer_test",
            cut_off_energy=250.0,
            kspacing=0.4,
            xc_functional="PBE",
            task="SinglePoint",
        )
        
        # Create DFTStaticLabelling with dimer settings
        dft_labelling = DFTStaticLabelling(
            static_energy_maker=castep_maker,
            dimer=True,
            dimer_box=[15.0, 15.5, 16.0],
            dimer_range=[1.5, 2.0],
            dimer_num=3,
        )
        
        # Test dimer configuration
        assert dft_labelling.dimer is True
        assert dft_labelling.dimer_box == [15.0, 15.5, 16.0]
        assert dft_labelling.dimer_range == [1.5, 2.0]
        assert dft_labelling.dimer_num == 3
        assert isinstance(dft_labelling.static_energy_maker, BaseCastepMaker)
    
    def test_dft_labelling_with_separate_isolated_maker(self):
        """Test DFTStaticLabelling with separate CASTEP maker for isolated atoms."""
        # Create test structure
        c_ase = Atoms("C", positions=[[0, 0, 0]], cell=[8, 8, 8], pbc=True)
        c_pmg = AseAtomsAdaptor.get_structure(c_ase)
        
        # Create different CASTEP makers for bulk and isolated
        bulk_maker = BaseCastepMaker(
            name="bulk_castep",
            cut_off_energy=300.0,
            kspacing=0.3,
            xc_functional="PBE",
            task="SinglePoint",
        )
        
        isolated_maker = BaseCastepMaker(
            name="isolated_castep",
            cut_off_energy=200.0,
            kspacing=100.0,  # Gamma-point only for isolated atoms
            xc_functional="PBE",
            task="SinglePoint",
        )
        
        # Create DFTStaticLabelling with both makers
        dft_labelling = DFTStaticLabelling(
            static_energy_maker=bulk_maker,
            static_energy_maker_isolated_atoms=isolated_maker,
            isolated_atom=True,
            e0_spin=True,
            isolatedatom_box=[18.0, 19.0, 20.0],
        )
        
        # Test configuration
        assert dft_labelling.static_energy_maker == bulk_maker
        assert dft_labelling.static_energy_maker_isolated_atoms == isolated_maker
        assert dft_labelling.isolated_atom is True
        assert dft_labelling.e0_spin is True
        assert dft_labelling.isolatedatom_box == [18.0, 19.0, 20.0]
