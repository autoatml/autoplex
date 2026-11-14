```{toctree}
:caption: User Guide
:hidden:
user/index
user/setup
user/tutorials
```

```{toctree}
:caption: Reference
:hidden:
reference/index
```

```{toctree}
:caption: Contributing Guide
:hidden:
dev/contributing
dev/dev_install
```

```{toctree}
:caption: About
:hidden:
about/changelog
about/contributors
about/license
```

![autoplex documentation](_static/autoplex_logo.png)
# autoplex documentation


**Date**: {sub-ref}`today`

**Useful links**:
[Source Repository](https://github.com/JaGeo/autoplex) |
[Issues & Ideas](https://github.com/JaGeo/autoplex/issues) |

`autoplex` is a software tool for the automated generation and benchmarking of machine learning (ML)-based interatomic potentials. 
The aim of autoplex is to provide a fully automated solution for creating high-quality ML potentials. 
The software is interfaced to multiple different ML potential fitting frameworks and to the atomate2 and ase environment 
for efficient high-throughput computations. 
The vision of this project is to allow a wide community of researchers to create accurate and reliable ML potentials for materials simulations.

::::{grid} 1 1 2 2
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: user/setup
:link-type: doc
:class-header: bg-light
**Quick Start**
^^^
The user guide provides information for getting started with *autoplex*.
:::

:::{grid-item-card}
:link: user/tutorials
:link-type: doc
:class-header: bg-light
**Tutorials**
^^^
Tutorials for using autoplex.
:::

:::{grid-item-card}
:class-header: bg-light
**Automation Setup**
^^^
Here you can find the setup guides for using `autoplex` with [MongoDB](user/mongodb.md), [FireWorks](user/mongodb.md#fireworks-configuration) and [jobflow-remote](user/jobflowremote.md).
:::

:::{grid-item-card}
:link: dev/dev_install
:link-type: doc
:class-header: bg-light
**Contributing Guide**
^^^
Do you want to develop your own workflows or improve existing functionalities?
Check out the Contributing Guide.
:::
::::
