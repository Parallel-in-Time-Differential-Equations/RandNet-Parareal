# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

The code was run using python 3.10. To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...


## Results

The code and outputs to fully replicate the tables and figures in the paper are provided. We used the following naming convention to organize the files:
- `PDE_name/`: folder containing simulation outputs. Note that for bigger PDEs the final solution $\boldsymbol{u}$ can be several hundred megabytes. In that case, we have removed it. We left a trace of this post-processing in the corresponding script.
- `PDE_name.bash`: bash script to run the simulations on the cluster. It requires MPI and SLURM. Nevertheless, the actual Python code can be easily adapted to run sequentially on any hardware.
- `PDE_name.py`': Python script to carry out the simulations. If you wish to run this sequentially, because you do not have easy access to MPI, remove the `mpi4py` import at the top, and remove the `pool=pool, parall='mpi'` arguments of the `run` method. Warning: expect out-of-memory issues and long (or very long) runtimes for bigger systems.
- `PDE_name_analysis.py`: Python script to replicate the results reported on the paper.

Execute the bash scripts as `bash Burgers.bash 128`, where 128 is the number of cores you wish to use; see the bash script for more information. For the Diffusion-Reaction equation, only the $N=512$ case is provided, by calling `bash ReactDiff.bash 512`. For the others, the values of `dx` need to be updates inside the bash script.

`Burgers_sequential.ipynb` provides an example of how to simulate Burgers' equation with 128 space discretizations that can run on any hardware. 

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Acknowledgments

Simulations for the shallow water equation have been carried out using code from the [PararealML](https://github.com/ViktorC/PararealML) GitHub repository with minor modifications. The cloned and edited code is located in `pararealml/`. My own code is shared using the same permissive licence.

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 