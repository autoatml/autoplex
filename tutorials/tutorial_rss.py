#!/usr/bin/env python
# coding: utf-8

# In[1]:


# We need this to run the tutorial directly in the jupyter notebook
ref_paths = {
"static_bulk_0": "vasp/rss_Si_small/static_bulk_0",
"static_bulk_1": "vasp/rss_Si_small/static_bulk_1",
"static_bulk_2": "vasp/rss_Si_small/static_bulk_2",
"static_bulk_3": "vasp/rss_Si_small/static_bulk_3",
"static_bulk_4": "vasp/rss_Si_small/static_bulk_4",
"static_bulk_5": "vasp/rss_Si_small/static_bulk_5",
"static_bulk_6": "vasp/rss_Si_small/static_bulk_6",
"static_bulk_7": "vasp/rss_Si_small/static_bulk_7",
"static_bulk_8": "vasp/rss_Si_small/static_bulk_8",
"static_bulk_9": "vasp/rss_Si_small/static_bulk_9",
"static_bulk_10": "vasp/rss_Si_small/static_bulk_10",
"static_bulk_11": "vasp/rss_Si_small/static_bulk_11",
"static_bulk_12": "vasp/rss_Si_small/static_bulk_12",
"static_bulk_13": "vasp/rss_Si_small/static_bulk_13",
"static_bulk_14": "vasp/rss_Si_small/static_bulk_14",
"static_bulk_15": "vasp/rss_Si_small/static_bulk_15",
"static_bulk_16": "vasp/rss_Si_small/static_bulk_16",
"static_bulk_17": "vasp/rss_Si_small/static_bulk_17",
"static_bulk_18": "vasp/rss_Si_small/static_bulk_18",
"static_bulk_19": "vasp/rss_Si_small/static_bulk_19",
"static_isolated_0":"vasp/rss_Si_small/static_isolated_0",
}


# We are using a config file. You can take a look at it here:

# In[2]:


from autoplex.settings import RssConfig
rss_config = RssConfig.from_file('autoplex/tutorials/rss_si_config.yaml')
print(rss_config)

# Now, we use this configuration to start the RSSMaker. We have chose a very small number of structures here. In reality, you would need many hundreds of calculation per genertion.
# In[3]:


from autoplex.auto.rss.flows import RssMaker
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.vasp.powerups import update_vasp_custodian_handlers
from jobflow import Flow


rss_job = RssMaker(name="rss", rss_config=rss_config).make()

autoplex_flow = update_user_incar_settings(Flow(jobs=[rss_job], output=rss_job.output), {"NPAR": 4})

autoplex_flow = update_vasp_custodian_handlers(autoplex_flow, custom_handlers={})

autoplex_flow.name = "new test rss"


# In[4]:


from mock_vasp import mock_vasp
from jobflow import run_locally

with mock_vasp(ref_paths=ref_paths, clean_folders=True) as mf:
    run_locally(
        autoplex_flow,
        create_folders=True,
        ensure_success=True,
        raise_immediately=True,
    )

