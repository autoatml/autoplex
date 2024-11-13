Tutorials
==========

*`autoplex` tutorials written by [autoplex](https://github.com/autoatml/autoplex) developers.*

The user is advised to have a general familiarity with the following software packages and tools:
 * VASP
 * the machine-learned interatomic potential (MLIP) framework that is aimed to be used
 * the [Materials Project](https://next-gen.materialsproject.org/) framework
 * atomate2
 * ase
 * basic Python knowledge

The tutorials are aimed to demonstrate the usage of `autoplex` to generate MLIPs and benchmark it to DFT results.
By the end of these tutorials, you should be able to:

* use `autoplex` out of the box with default settings
* customize the workflow settings
* use any of the submodule workflows to build your own workflows

## Tutorial table of content

```{toctree}
:maxdepth: 2
quickstart/quickstart
quickstart/installation
```

```{toctree}
:caption: Random structure searching (RSS) workflow
:maxdepth: 3
rss/rss
```

```{toctree}
:caption: Phonon-accurate machine-learned potentials workflow
:maxdepth: 3
phonon/flows/flows
phonon/flows/generation/data
phonon/flows/fitting/fitting
phonon/flows/benchmark/benchmark
```

