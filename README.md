# G1 Mujoco Gymnasium Environment
This repository contains a recreation of the [MJLab](https://github.com/mujocolab/mjlab.git) Mjlab-Velocity-Flat-Unitree-G1 environment with plain MuJoCo.
This was done for use with incremental/online learning algorithms that require fast single-environment iteration, where the parallel system offered by MJLab was far too slow.


## Installation

To install your new environment, run the following commands:

```{shell}
cd g1_mujoco_env
pip install -e .
```

