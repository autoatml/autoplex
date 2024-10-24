Setup
================

We are referring the user to the [installation guide of atomate2](https://materialsproject.github.io/atomate2/user/install.html) in order to setup the mandatory prerequisites to 
be able to use `autoplex`.

After setting up `atomate2`, make sure to add `VASP_INCAR_UPDATES: {"NPAR": number}` in your ~/atomate2/config/atomate2.yaml file. 
Set a number that is a divisor of the number of tasks you use for the VASP calculations.

You can manage your `autoplex` workflow using [`FireWorks`](https://materialsproject.github.io/fireworks/) or [`jobflow-remote`](https://matgenix.github.io/jobflow-remote/). 
Please follow the installation and setup instructions on the respective guide website.
Both packages rely on the [MongoDB](https://www.mongodb.com/) database manager for data storage.

We recommend using `jobflow-remote` as it is more flexible to use, especially on clusters where users cannot store their
own MongoDB. You can find a more comprehensive `jobflow-remote` tutorial [here](jobflowremote.md).