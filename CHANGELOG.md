# Changelog

## v0.1.4

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### 🚀 New Features
* Improve GAP fitting by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/398
* Improve usability of RSS workflows by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/402
### 📚 Documentation
* Improve the documentation for the RSS workflow by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/404
### 🔄 Other Changes
* Generalize data creation for phonon workflow and beyond by @JaGeo in https://github.com/autoatml/autoplex/pull/390


**Full Changelog**: https://github.com/autoatml/autoplex/compare/v0.1.3...v0.1.4


## v0.1.3

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### 🚀 New Features
* Enable NEP interface by @naik-aakash in https://github.com/autoatml/autoplex/pull/288
* RSS Test and Interactive Tutorial for Phonon and RSS Workflow by @JaGeo in https://github.com/autoatml/autoplex/pull/343
* Move Maker defaults to CompleteDFTvsMLBenchmarkWorkflow definition by @QuantumChemist in https://github.com/autoatml/autoplex/pull/333
### 🐛 Bug Fixes
* Regularization in higher dimensions by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/354
### 📚 Documentation
* Include executable tutorials in documentation website by @naik-aakash in https://github.com/autoatml/autoplex/pull/349
* New additions to documentation by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/363
* Update README.md by @JaGeo in https://github.com/autoatml/autoplex/pull/366
### 🧹 House-Keeping
* add non executed tutorial notebooks (reduce size) by @naik-aakash in https://github.com/autoatml/autoplex/pull/348
* Add MANIFEST.in by @naik-aakash in https://github.com/autoatml/autoplex/pull/357
* update dependencies by @naik-aakash in https://github.com/autoatml/autoplex/pull/367


**Full Changelog**: https://github.com/autoatml/autoplex/compare/v0.1.2...v0.1.3


## v0.1.2

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### 🚀 New Features
* Adapt Makers to use pydantic models instead of yamls by @naik-aakash in https://github.com/autoatml/autoplex/pull/307
### 📚 Documentation
* small docs extension by @QuantumChemist in https://github.com/autoatml/autoplex/pull/332
### 🧹 House-Keeping
* Update dependencies & CI publish workflow by @naik-aakash in https://github.com/autoatml/autoplex/pull/334


**Full Changelog**: https://github.com/autoatml/autoplex/compare/v0.1.1...v0.1.2

## v0.1.1
### What's changed

* Update devcontainer.json version tag by @JaGeo in https://github.com/autoatml/autoplex/pull/303
* Update start.md by @JaGeo in https://github.com/autoatml/autoplex/pull/304
* add autoplex preprint to readme by @JaGeo in https://github.com/autoatml/autoplex/pull/309
* Update README.md by @JaGeo in https://github.com/autoatml/autoplex/pull/310
* Update README.md by @JaGeo in https://github.com/autoatml/autoplex/pull/311
* Update docs dependencies by @naik-aakash in https://github.com/autoatml/autoplex/pull/315
* Update test durations file by @JaGeo in https://github.com/autoatml/autoplex/pull/318
* Revert "Update test durations file" by @naik-aakash in https://github.com/autoatml/autoplex/pull/319
* pre-commit autoupdate by @pre-commit-ci in https://github.com/autoatml/autoplex/pull/321
* Build an iterative phonon flow by @JaGeo in https://github.com/autoatml/autoplex/pull/306
* Update test durations file by @JaGeo in https://github.com/autoatml/autoplex/pull/325
* Allow fitting procedure on a different cluster and make the fitting database accessible via the MongoDB  by @JaGeo in https://github.com/autoatml/autoplex/pull/314
* Update test durations file by @JaGeo in https://github.com/autoatml/autoplex/pull/326
* Fix various smaller bugs (including correct counting of the random structure seed in case very similar input structures are chosen)

**Full Changelog**: https://github.com/autoatml/autoplex/compare/v0.1.0...v0.1.1

## v0.1.0
### What's Changed
* Update devcontainer.json version tag by @JaGeo in https://github.com/autoatml/autoplex/pull/279
* Fix Docs build trigger case by @naik-aakash in https://github.com/autoatml/autoplex/pull/283
* Bump hiphive from 1.3.1 to 1.4 by @dependabot in https://github.com/autoatml/autoplex/pull/285
* pre-commit autoupdate by @pre-commit-ci in https://github.com/autoatml/autoplex/pull/286
* Bump mace-torch from 0.3.8 to 0.3.9 by @dependabot in https://github.com/autoatml/autoplex/pull/290
* Refactoring of the MLIP modules and other non-urgent things by @QuantumChemist in https://github.com/autoatml/autoplex/pull/280
* Update test durations file by @JaGeo in https://github.com/autoatml/autoplex/pull/294
* add more diversity to rattled cell creation by @JaGeo in https://github.com/autoatml/autoplex/pull/298

### New Contributors
* @pre-commit-ci made their first contribution in https://github.com/autoatml/autoplex/pull/286

**Full Changelog**: https://github.com/autoatml/autoplex/compare/v0.0.9...v0.1.0

## v0.0.9
### What's Changed
The release is in principle same as v0.0.8 with following changes: LAMMPS is now pinned to a stable release (`stable_29Aug2024_update1`):
* Pin lammps ci and in Docs by @naik-aakash in https://github.com/autoatml/autoplex/pull/277
* Update devcontainer.json version tag by @JaGeo in https://github.com/autoatml/autoplex/pull/276

**Full Changelog**: https://github.com/autoatml/autoplex/compare/v0.0.8...v0.0.9

## v0.0.8
### What's Changed
* Update devcontainer.json version tag by @JaGeo in https://github.com/autoatml/autoplex/pull/253
* Fix recursive autoupdate durations by @naik-aakash in https://github.com/autoatml/autoplex/pull/255
* Fix MLIP related issues with the benchmark results file by @QuantumChemist in https://github.com/autoatml/autoplex/pull/243
* ignore type checking block by @naik-aakash in https://github.com/autoatml/autoplex/pull/256
* Bump autodoc-pydantic from 2.0.1 to 2.2.0 by @dependabot in https://github.com/autoatml/autoplex/pull/260
* Make sure that supercell_matrices of single-atom-displaced and rattled supercells are the same per default by @QuantumChemist in https://github.com/autoatml/autoplex/pull/258
* Update test durations file by @JaGeo in https://github.com/autoatml/autoplex/pull/265
* remove redundant else case by @naik-aakash in https://github.com/autoatml/autoplex/pull/266
* Updated Documentation for Autoplex, Including RSS Features by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/226
* Bump mace-torch from 0.3.7 to 0.3.8 by @dependabot in https://github.com/autoatml/autoplex/pull/261
* Bump numpydoc from 1.6.0 to 1.8.0 by @dependabot in https://github.com/autoatml/autoplex/pull/268
* Bump lightning-utilities from 0.11.2 to 0.11.9 by @dependabot in https://github.com/autoatml/autoplex/pull/269
* Implement error handling for fit error metrics plots by @QuantumChemist in https://github.com/autoatml/autoplex/pull/272
* Raise informative errors on missing non-python dependencies when invoked by @naik-aakash in https://github.com/autoatml/autoplex/pull/273

**Full Changelog**: https://github.com/autoatml/autoplex/compare/v0.0.7...v0.0.8

## v0.0.7
### What's Changed
* Clean up pyproject.toml to enable a strict/non-strict installation procedure by @JaGeo in https://github.com/autoatml/autoplex/pull/242
* Misc CI workflow improvements by @naik-aakash in https://github.com/autoatml/autoplex/pull/238
* Update devcontainer.json version tag by @JaGeo in https://github.com/autoatml/autoplex/pull/245
* Update test durations file by @JaGeo in https://github.com/autoatml/autoplex/pull/244
* Improve compatibility with remote clusters by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/247


**Full Changelog**: https://github.com/autoatml/autoplex/compare/v0.0.5...v0.0.6

## v0.0.6
 * does not exist due to a release error

## v0.0.5
Bug fix for a missing file export in v0.0.4 (RSS functionality)

## v0.0.4
Essentially, it is the same version as v0.0.2. The release has, however, been updated to work with stable atomate2 and mace-torch versions.

## v0.0.2
### What's Changed
* fix m3gnet issue by @naik-aakash in https://github.com/autoatml/autoplex/pull/76
* Docstrings, type-hints, documentation fixes by @QuantumChemist in https://github.com/autoatml/autoplex/pull/78
* Test buildcell by @JaGeo in https://github.com/autoatml/autoplex/pull/82
* installation of buildcell by @JaGeo in https://github.com/autoatml/autoplex/pull/81
* Merging RSS code by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/69
* Update README.md by @JaGeo in https://github.com/autoatml/autoplex/pull/85
* [WIP] Integration and test of RSS code by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/84
* RSS fixes by @QuantumChemist in https://github.com/autoatml/autoplex/pull/89
* Hookean tests by @MorrowChem in https://github.com/autoatml/autoplex/pull/92
* Run full pre-commit hook in CI by @QuantumChemist in https://github.com/autoatml/autoplex/pull/95
* Implementation of adaptive supercell settings and improvement of the documentation by @QuantumChemist in https://github.com/autoatml/autoplex/pull/80
* Supercell Tests by @JaGeo in https://github.com/autoatml/autoplex/pull/98
* Miscellaneous changes by @QuantumChemist in https://github.com/autoatml/autoplex/pull/102
* Merging by @QuantumChemist in https://github.com/autoatml/autoplex/pull/103
* Add devcontainer by @naik-aakash in https://github.com/autoatml/autoplex/pull/96
* string python versions and fix repo name by @naik-aakash in https://github.com/autoatml/autoplex/pull/106
* remove accidental - from docker workflow by @naik-aakash in https://github.com/autoatml/autoplex/pull/107
* Update devcontainer image and docker-publish workflow by @naik-aakash in https://github.com/autoatml/autoplex/pull/109
* Update Dockerfile to use micromamba and make it uv ready by @naik-aakash in https://github.com/autoatml/autoplex/pull/110
* fix linting for Dockerfile by @naik-aakash in https://github.com/autoatml/autoplex/pull/111
* Optimize test wf by @naik-aakash in https://github.com/autoatml/autoplex/pull/112
* Fix quippy error in CI by @naik-aakash in https://github.com/autoatml/autoplex/pull/115
* fix final coverage report not being generated by @naik-aakash in https://github.com/autoatml/autoplex/pull/117
* Update copyright notice for reused code by @MorrowChem in https://github.com/autoatml/autoplex/pull/120
* Update LICENSE by @JaGeo in https://github.com/autoatml/autoplex/pull/121
* Update docker-publish.yml by @naik-aakash in https://github.com/autoatml/autoplex/pull/126
* pin atomate2 to last working commit by @naik-aakash in https://github.com/autoatml/autoplex/pull/127
* Update devcontainer & fix linting by @naik-aakash in https://github.com/autoatml/autoplex/pull/129
* Prettify tests by @naik-aakash in https://github.com/autoatml/autoplex/pull/130
* Revert prettify tests by @naik-aakash in https://github.com/autoatml/autoplex/pull/132
* Fix force field dependencies by @naik-aakash in https://github.com/autoatml/autoplex/pull/134
* Fix liniting error & Update test durations by @naik-aakash in https://github.com/autoatml/autoplex/pull/135
* Merge main into sc algo by @QuantumChemist in https://github.com/autoatml/autoplex/pull/138
* Bump sphinx from 7.2.6 to 8.0.2 by @dependabot in https://github.com/autoatml/autoplex/pull/123
* merge main into sc algo by @QuantumChemist in https://github.com/autoatml/autoplex/pull/139
* Revert "Bump sphinx from 7.2.6 to 8.0.2" by @naik-aakash in https://github.com/autoatml/autoplex/pull/143
* merging main to adapt_sc_algo for docs fixes by @QuantumChemist in https://github.com/autoatml/autoplex/pull/154
* Copyright notice by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/155
* Fix doc by @naik-aakash in https://github.com/autoatml/autoplex/pull/157
* Fix doc by @naik-aakash in https://github.com/autoatml/autoplex/pull/159
* Include autoplex docker by @naik-aakash in https://github.com/autoatml/autoplex/pull/161
* Enable more refined supercell settings and adapt DFT settings in phonon workflow by @JaGeo in https://github.com/autoatml/autoplex/pull/100
* Docs fixes by @QuantumChemist in https://github.com/autoatml/autoplex/pull/162
* Add docs deployment to ci by @naik-aakash in https://github.com/autoatml/autoplex/pull/73
* added the current authors of the documentation to the conf file by @QuantumChemist in https://github.com/autoatml/autoplex/pull/164
* Update Dockerfile by @naik-aakash in https://github.com/autoatml/autoplex/pull/166
* Fix tests by @naik-aakash in https://github.com/autoatml/autoplex/pull/163
* Update install instructions and MISC cleanup by @naik-aakash in https://github.com/autoatml/autoplex/pull/168
* Adjust glue xml file path handling by @QuantumChemist in https://github.com/autoatml/autoplex/pull/169
* Small changes in README and pyproject files by @QuantumChemist in https://github.com/autoatml/autoplex/pull/170
* Update index.md by @naik-aakash in https://github.com/autoatml/autoplex/pull/172
* Update split durations by @QuantumChemist in https://github.com/autoatml/autoplex/pull/171
* Small cleanup in docs conf file and added a favicon for the documentation pages by @QuantumChemist in https://github.com/autoatml/autoplex/pull/174
* update colors to fix dark mode by @naik-aakash in https://github.com/autoatml/autoplex/pull/177
* add PyPi publish workflow by @naik-aakash in https://github.com/autoatml/autoplex/pull/152
* Manual supercells by @QuantumChemist in https://github.com/autoatml/autoplex/pull/176
* remove hardcoded json file name by @naik-aakash in https://github.com/autoatml/autoplex/pull/185
* clear cache in docker image & test workflow by @naik-aakash in https://github.com/autoatml/autoplex/pull/189
* Added "datatype" to results summary file by @QuantumChemist in https://github.com/autoatml/autoplex/pull/192
* adjusting VASP/jobs settings by @QuantumChemist in https://github.com/autoatml/autoplex/pull/200
* Update README.md by @JaGeo in https://github.com/autoatml/autoplex/pull/201
* add tags to docker images and cleanup workflow by @naik-aakash in https://github.com/autoatml/autoplex/pull/206
* Fix dockerbuild workflow by @naik-aakash in https://github.com/autoatml/autoplex/pull/207
* Cleanup ghcr by @naik-aakash in https://github.com/autoatml/autoplex/pull/208
* update image urls by @naik-aakash in https://github.com/autoatml/autoplex/pull/209
* Add workflow to autoupdate devcontainer.json version tags on new release by @naik-aakash in https://github.com/autoatml/autoplex/pull/212
* Make the MACE fit more flexible and build a finetuning workflow by @JaGeo in https://github.com/autoatml/autoplex/pull/182
* Update docker image - Include LAMMPS  by @naik-aakash in https://github.com/autoatml/autoplex/pull/210
* Auto update test durations by @naik-aakash in https://github.com/autoatml/autoplex/pull/215
* Revise lammps compilation by @naik-aakash in https://github.com/autoatml/autoplex/pull/217
* Update README.md with new RSS functionalities and clean up by @JaGeo in https://github.com/autoatml/autoplex/pull/202
* fix linting by @JaGeo in https://github.com/autoatml/autoplex/pull/218
* Make version number consistent by @JaGeo in https://github.com/autoatml/autoplex/pull/220
* raise pr via update_devcontainer.yml by @naik-aakash in https://github.com/autoatml/autoplex/pull/221
* Created a new unified flow module for RSS. by @YuanbinLiu in https://github.com/autoatml/autoplex/pull/203
* move to src layout by @naik-aakash in https://github.com/autoatml/autoplex/pull/224

### New Contributors
* @dependabot made their first contribution in https://github.com/autoatml/autoplex/pull/123

**Full Changelog**: https://github.com/autoatml/autoplex/compare/v0.0.1...v0.0.2

## v0.0.1
* First internal release of autoplex
* Added initial workflows for phonon accurate automated gap fits
* Added interface for MACE, M3GNET, J-ACE, NEQUIP potential fits
* Added option to include rattled structures in MLIP fits
