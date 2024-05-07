#!/bin/bash
file=$(date +%s)
cp "/home/maths/strkss/massi/elm/Burgers_perf_across_m.py" "/home/maths/strkss/massi/elm/Burgers_perf_across_m_"$file".py"
	sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 
module load GCCcore/11.3.0 Python/3.10.4
# module load SciPy-bundle/2022.05
# module load matplotlib/3.5.2
# #pip install --upgrade pip
# #pip install spyder-kernels

# #pip install spyder-kernels==2.3.1
# #pip install ipython_genutils==0.2.0
# #pip install jaxlib==0.4.6
# #pip install jax==0.4.6
cd "/home/maths/strkss/massi/elm"
source venv/bin/activate


# Sequential application
#srun python -u run.py

# Multiprocess application
srun python -u -m mpi4py.futures Burgers_perf_across_m_"$file".py

# The following command is used to open an interactive session
# salloc --ntasks-per-node=48 --nodes=3 --cpus-per-task=1 --time=9:00:00 --mem-per-cpu=3700

exit 0

EOT
