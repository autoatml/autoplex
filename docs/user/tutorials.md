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

## Contributions

- [Christina Ertural (christina.ertural@bam.de)](quickstart/quickstart.md)
- [Christina Ertural (christina.ertural@bam.de) and Aakash Naik (aakash.naik@bam.de)](quickstart/installation.md)
- [Yuanbin Liu (lyb122502@126.com)](rss/rss.md)
- [Christina Ertural (christina.ertural@bam.de) and Janine George (janine.george@bam.de)](phonon/flows/flows.md)
- [Christina Ertural (christina.ertural@bam.de)](phonon/flows/generation/data.md)
- [Christina Ertural (christina.ertural@bam.de)](phonon/flows/fitting/fitting.md)
- [Christina Ertural (christina.ertural@bam.de)](phonon/flows/benchmark/benchmark.md)
- [Aakash Naik (aakash.naik@bam.de) and Christina Ertural (christina.ertural@bam.de)](jobflowremote.md)
- [Aakash Naik (aakash.naik@bam.de) and Christina Ertural (christina.ertural@bam.de)](mongodb.md)

