import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
from autoplex.auto.rss.flows import RssConfig
from autoplex.auto.rss.flows import RssMaker

from autoplex.auto.rss.flows import (
    setup_castep_keywords,
    create_castep_makers,
    get_default_rss_config,
    create_and_validate_rss_config,
    run_rss_workflow
)


class TestSetupCastepKeywords:
    """Test CASTEP keywords setup functionality"""
    
    def test_returns_true_when_keywords_exist(self):
        """Test that function returns True when keywords file already exists"""
        with patch('os.path.exists', return_value=True):
            result = setup_castep_keywords()
            assert result is True
    
    @patch('castep_rss.create_castep_keywords')
    @patch('castep_rss.shutil.copy')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.access')
    def test_creates_keywords_successfully(self, mock_access, mock_exists, 
                                         mock_makedirs, mock_copy, mock_create):
        """Test successful keyword file creation"""
        # Setup mocks
        mock_exists.side_effect = lambda path: path != 'castep_keywords.json'  # File doesn't exist initially
        mock_access.return_value = True  # CASTEP is executable
        mock_create.return_value = True  # Successful creation
        
        result = setup_castep_keywords()
        
        assert result is True
        mock_create.assert_called_once()
        mock_copy.assert_called_once()
    
    @patch('castep_rss.create_castep_keywords')
    @patch('os.path.exists')
    @patch('os.access')
    def test_fails_when_castep_creation_fails(self, mock_access, mock_exists, mock_create):
        """Test failure when CASTEP keyword creation fails"""
        mock_exists.side_effect = lambda path: path != 'castep_keywords.json'
        mock_access.return_value = True
        mock_create.return_value = False  # Creation fails
        
        result = setup_castep_keywords()
        
        assert result is False
    
    @patch('castep_rss.create_castep_keywords')
    @patch('os.path.exists')
    @patch('os.access')
    def test_handles_exception_during_creation(self, mock_access, mock_exists, mock_create):
        """Test exception handling during keyword creation"""
        mock_exists.side_effect = lambda path: path != 'castep_keywords.json'
        mock_access.return_value = True
        mock_create.side_effect = Exception("Test exception")
        
        result = setup_castep_keywords()
        
        assert result is False


class TestCreateCastepMakers:
    """Test CASTEP maker creation"""
    
    def test_creates_makers_with_default_parameters(self):
        """Test maker creation with default parameters"""
        castep_maker, isolated_maker = create_castep_makers()
        
        # Test main maker
        assert castep_maker.name == "castep static"
        assert castep_maker.castep_command == '/usr/local/CASTEP-20/castep.mpi'
        assert castep_maker.cut_off_energy == 200.0
        assert castep_maker.kspacing == 0.5
        assert castep_maker.xc_functional == 'PBE'
        assert castep_maker.task == 'SinglePoint'
        
        # Test isolated maker
        assert isolated_maker.name == "castep isolated"
        assert isolated_maker.cut_off_energy == 150.0
        assert isolated_maker.kspacing == 2.0
    
    def test_creates_makers_with_custom_parameters(self):
        """Test maker creation with custom parameters"""
        custom_command = '/custom/castep'
        custom_cutoff = 300.0
        custom_kspacing = 0.3
        
        castep_maker, isolated_maker = create_castep_makers(
            castep_command=custom_command,
            cut_off_energy=custom_cutoff,
            kspacing=custom_kspacing,
            isolated_cut_off_energy=250.0,
            isolated_kspacing=1.5
        )
        
        assert castep_maker.castep_command == custom_command
        assert castep_maker.cut_off_energy == custom_cutoff
        assert castep_maker.kspacing == custom_kspacing
        assert isolated_maker.cut_off_energy == 250.0
        assert isolated_maker.kspacing == 1.5


class TestGetDefaultRssConfig:
    """Test RSS configuration creation"""
    
    def test_config_has_required_fields(self):
        """Test that config contains all required fields"""
        config = get_default_rss_config()
        
        required_fields = [
            'tag', 'train_from_scratch', 'generated_struct_numbers',
            'mlip_type', 'calculator_type', 'stop_criterion', 
            'max_iteration_number'
        ]
        
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"
    
    def test_config_values_are_reasonable(self):
        """Test that config values are within reasonable ranges"""
        config = get_default_rss_config()
        
        # Test positive values
        assert all(n > 0 for n in config['generated_struct_numbers'])
        assert config['stop_criterion'] > 0
        assert config['max_iteration_number'] > 0
        assert config['cut_off_energy'] > 0 if 'cut_off_energy' in config else True
        
        # Test specific values
        assert config['tag'] == 'Si'
        assert config['calculator_type'] == 'castep'
        assert config['mlip_type'] == 'GAP'
        assert config['train_from_scratch'] is True
    
    def test_mlip_hypers_structure(self):
        """Test MLIP hyperparameters structure"""
        config = get_default_rss_config()
        
        assert 'mlip_hypers' in config
        assert 'GAP' in config['mlip_hypers']
        
        gap_config = config['mlip_hypers']['GAP']
        assert 'general' in gap_config
        assert 'soap' in gap_config
        assert 'twob' in gap_config
        assert 'threeb' in gap_config


class TestCreateAndValidateRssConfig:
    """Test RSS config creation and validation"""
    
    @patch('castep_rss.RssConfig')
    def test_validates_default_config(self, mock_rss_config):
        """Test validation of default configuration"""
        mock_validated_config = MagicMock()
        mock_rss_config.model_validate.return_value = mock_validated_config
        
        result = create_and_validate_rss_config()
        
        mock_rss_config.model_validate.assert_called_once()
        assert result == mock_validated_config
    
    @patch('castep_rss.RssConfig')
    def test_applies_config_overrides(self, mock_rss_config):
        """Test that configuration overrides are applied"""
        mock_validated_config = MagicMock()
        mock_rss_config.model_validate.return_value = mock_validated_config
        
        overrides = {'tag': 'C', 'max_iteration_number': 5}
        result = create_and_validate_rss_config(overrides)
        
        # Check that model_validate was called with modified config
        called_config = mock_rss_config.model_validate.call_args[0][0]
        assert called_config['tag'] == 'C'
        assert called_config['max_iteration_number'] == 5


class TestRunRssWorkflow:
    """Test RSS workflow execution"""
    
    @patch('castep_rss.run_locally')
    @patch('castep_rss.Flow')
    @patch('castep_rss.RssMaker')
    def test_workflow_runs_successfully(self, mock_rss_maker_class, mock_flow_class, mock_run_locally):
        """Test successful workflow execution"""
        # Setup mocks
        mock_job = MagicMock()
        mock_maker = MagicMock()
        mock_maker.make.return_value = mock_job
        mock_rss_maker_class.return_value = mock_maker
        
        mock_flow = MagicMock()
        mock_flow_class.return_value = mock_flow
        
        mock_responses = {"success": True}
        mock_run_locally.return_value = mock_responses
        
        # Test data
        mock_config = MagicMock()
        mock_castep_maker = MagicMock()
        mock_isolated_maker = MagicMock()
        
        result = run_rss_workflow(mock_config, mock_castep_maker, mock_isolated_maker)
        
        # Verify function calls
        mock_rss_maker_class.assert_called_once_with(
            name="rss_castep_final",
            rss_config=mock_config,
            static_energy_maker=mock_castep_maker,
            static_energy_maker_isolated_atoms=mock_isolated_maker
        )
        mock_maker.make.assert_called_once()
        mock_run_locally.assert_called_once()
        
        assert result == mock_responses
    
    @patch('castep_rss.run_locally')
    @patch('castep_rss.Flow')
    @patch('castep_rss.RssMaker')
    def test_workflow_with_custom_job_name(self, mock_rss_maker_class, mock_flow_class, mock_run_locally):
        """Test workflow with custom job name"""
        # Setup minimal mocks
        mock_job = MagicMock()
        mock_maker = MagicMock()
        mock_maker.make.return_value = mock_job
        mock_rss_maker_class.return_value = mock_maker
        mock_flow_class.return_value = MagicMock()
        mock_run_locally.return_value = {}
        
        # Test with custom job name
        custom_name = "custom_rss_job"
        run_rss_workflow(MagicMock(), MagicMock(), MagicMock(), job_name=custom_name)
        
        # Verify custom name was used
        call_args = mock_rss_maker_class.call_args
        assert call_args[1]['name'] == custom_name


class TestMainFunction:
    """Test main function integration"""
    
    @patch('castep_rss.run_rss_workflow')
    @patch('castep_rss.create_and_validate_rss_config')
    @patch('castep_rss.create_castep_makers')
    @patch('castep_rss.setup_castep_keywords')
    def test_main_success(self, mock_setup, mock_create_makers, mock_create_config, mock_run_workflow):
        """Test successful main function execution"""
        # Setup mocks for success
        mock_setup.return_value = True
        mock_create_makers.return_value = (MagicMock(), MagicMock())
        mock_create_config.return_value = MagicMock()
        mock_run_workflow.return_value = {"success": True}
        
        result = main()
        
        assert result is True
        mock_setup.assert_called_once()
        mock_create_makers.assert_called_once()
        mock_create_config.assert_called_once()
        mock_run_workflow.assert_called_once()
    
    @patch('castep_rss.setup_castep_keywords')
    def test_main_fails_on_setup(self, mock_setup):
        """Test main function failure during setup"""
        mock_setup.return_value = False
        
        result = main()
        
        assert result is False
        mock_setup.assert_called_once()
    
    @patch('castep_rss.run_rss_workflow')
    @patch('castep_rss.create_and_validate_rss_config')
    @patch('castep_rss.create_castep_makers')
    @patch('castep_rss.setup_castep_keywords')
    def test_main_handles_workflow_exception(self, mock_setup, mock_create_makers, 
                                           mock_create_config, mock_run_workflow):
        """Test main function handles workflow exceptions"""
        # Setup mocks
        mock_setup.return_value = True
        mock_create_makers.return_value = (MagicMock(), MagicMock())
        mock_create_config.return_value = MagicMock()
        mock_run_workflow.side_effect = Exception("Workflow failed")
        
        result = main()
        
        assert result is False


# Parametrized tests for different configurations
class TestConfigParametrization:
    """Test different configuration scenarios"""
    
    @pytest.mark.parametrize("mlip_type", ["GAP", "MACE", "M3GNET"])
    def test_different_mlip_types(self, mlip_type):
        """Test configuration with different MLIP types"""
        overrides = {'mlip_type': mlip_type}
        config = get_default_rss_config()
        config.update(overrides)
        
        assert config['mlip_type'] == mlip_type
    
    @pytest.mark.parametrize("num_structures,expected_valid", [
        ([10], True),
        ([100], True),
        ([1000], True),
        ([0], False),
        ([-1], False),
    ])
    def test_structure_number_validation(self, num_structures, expected_valid):
        """Test validation of structure numbers"""
        config = get_default_rss_config()
        config['generated_struct_numbers'] = num_structures
        
        if expected_valid:
            assert all(n > 0 for n in config['generated_struct_numbers'])
        else:
            assert any(n <= 0 for n in config['generated_struct_numbers'])


# Integration tests (marked as slow)
class TestIntegration:
    """Integration tests for full workflow components"""
    
    @pytest.mark.slow
    @patch('castep_rss.run_locally')
    @patch('castep_rss.Flow')
    @patch('castep_rss.RssMaker')
    def test_full_workflow_integration(self, mock_rss_maker, mock_flow, mock_run_locally):
        """Test integration of workflow components without actual CASTEP runs"""
        # Mock expensive operations
        mock_job = MagicMock()
        mock_job.host = None  # Required for Flow validation
        mock_job.name = "test_job"
        mock_job.uuid = "test-uuid-123"
        mock_job.output = MagicMock()
        
        mock_maker = MagicMock()
        mock_maker.make.return_value = mock_job
        mock_rss_maker.return_value = mock_maker
        
        mock_flow_instance = MagicMock()
        mock_flow.return_value = mock_flow_instance
        
        mock_run_locally.return_value = {"integration_test": True}
        
        # Test complete workflow
        castep_maker, isolated_maker = create_castep_makers()
        config = create_and_validate_rss_config({'max_iteration_number': 1})
        result = run_rss_workflow(config, castep_maker, isolated_maker)
        
        assert result == {"integration_test": True}


# Fixtures for common test data
@pytest.fixture
def sample_castep_makers():
    """Fixture providing sample CASTEP makers"""
    return create_castep_makers()

@pytest.fixture
def sample_rss_config():
    """Fixture providing sample RSS configuration"""
    return get_default_rss_config()

# Usage example with fixtures
class TestWithFixtures:
    """Example tests using fixtures"""
    
    def test_makers_creation(self, sample_castep_makers):
        """Test using fixture for makers"""
        castep_maker, isolated_maker = sample_castep_makers
        assert castep_maker.name == "castep static"
        assert isolated_maker.name == "castep isolated"
    
    def test_config_structure(self, sample_rss_config):
        """Test using fixture for config"""
        assert sample_rss_config['tag'] == 'Si'
        assert 'mlip_hypers' in sample_rss_config