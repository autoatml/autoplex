---
orphan: true
---

# RSS relaxation safeguards

RSS relaxations validate each trial structure before calling the interatomic
potential. These checks also apply inside optimizer line searches, where a
single potential evaluation can otherwise run indefinitely even when
`max_steps` is small. One-atom periodic structures remain supported; periodic
self images are included in the minimum-distance check.

Configure the safeguards in `rss_config.yaml`:

```yaml
rss_safety:
  max_walltime: 600.0              # seconds per structure, including model loading
  min_volume_per_atom: 0.1        # A^3, fully periodic cells only
  min_distance: 0.1               # A, atom pairs and periodic self images
  max_cell_condition_number: 10000.0
```

These are the defaults. The geometry limits are conservative guards against
catastrophic collapse, not a substitute for a physically appropriate repulsive
potential. Set system-specific limits when needed. `min_volume_per_atom` and
`min_distance` may be `null` to disable those particular checks. Finite
coordinates and an independent, sufficiently well-conditioned periodic lattice
are always required. Nonperiodic zero cell vectors are supported.

Each structure runs in a separate supervised process, up to
`num_processes_rss` processes at once. The supervisor detects worker exit
without a result and terminates calculations that exceed `max_walltime`,
including native code that does not return to Python. The limit starts when a
child is launched; time waiting for a free worker does not count. It also
applies when `num_processes_rss: 1`. On platforms using multiprocessing spawn,
put the workflow invocation under `if __name__ == "__main__":`.

Invalid geometries, crashes, and timeouts produce warnings identifying the
structure. Successful trajectory paths retain their original order; failed
entries remain `None`. An RSS group with no converged structures fails with a
clear error. Selection also fails explicitly when fewer trajectory frames than
requested are available. Inspect the failure warnings before increasing limits.

The wall-time limit does not impose a hard memory limit. Use operating-system
or scheduler memory limits to protect the host; the supervisor handles workers
killed by those limits. A negative exit code is reported as such and is not
assumed to prove an OOM event.

Capture both logging streams, for example:

```bash
python -u rss_workflow.py > rss_workflow.log 2>&1
```
