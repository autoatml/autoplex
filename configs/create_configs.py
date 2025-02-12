"""Example script to create hyperparameters and rss config objects using json/yaml files."""

from autoplex.settings import MLIPHypers, RssConfig

# create a custom hyperparameters object using json file
custom_hyperparameters = MLIPHypers.from_file("mlip_hypers.json")

# create a custom rss config object using json file
custom_rss_config = RssConfig.from_file(
    "rss_config.yaml"
)  # rss_config_all.yaml contains list of all hypers and supported MLIPs
