# RSS Workflow


```python
import os

os.environ["OMP_NUM_THREADS"] = "1"
```

We will not perform VASP calculations in realtime for this tutorial, but rather mock vasp runs.
Thus, it is necessary to set folders with pre-computed VASP output files for execution in the notebook.



```python
ref_paths = {}
base_path = "vasp/rss_Si_small/"

for i in range(20):
    ref_paths[f"static_bulk_{i}"] = f"{base_path}static_bulk_{i}"

ref_paths["static_isolated_0"] = f"{base_path}static_isolated_0"
```

We are using a config file to initialize the parameters of the workflow. You can take a look at it [here](https://github.com/autoatml/autoplex/blob/main/tutorials/rss_si_config.yaml)


```python
from autoplex.settings import RssConfig

rss_config = RssConfig.from_file("rss_si_config.yaml")
print(rss_config)
```

Now, we use this configuration to start the RSSMaker. We have chose a very small number of structures here. In reality, you would need many hundreds of calculation per genertion.


```python
import warnings

from atomate2.vasp.powerups import update_user_incar_settings, update_vasp_custodian_handlers
from jobflow import Flow

from autoplex.auto.rss.flows import RssMaker

warnings.filterwarnings("ignore")


rss_job = RssMaker(name="rss", rss_config=rss_config).make()

autoplex_flow = update_user_incar_settings(Flow(jobs=[rss_job], output=rss_job.output), {"NPAR": 4})

autoplex_flow = update_vasp_custodian_handlers(autoplex_flow, custom_handlers={})

autoplex_flow.name = "new test rss"
```


```python
from jobflow import run_locally
from mock_vasp import mock_vasp

with mock_vasp(ref_paths=ref_paths, clean_folders=True) as mf:
    run_locally(
        autoplex_flow,
        create_folders=True,
        ensure_success=True,
        raise_immediately=True,
    )
```
