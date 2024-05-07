#!/bin/bash
if [ $1 -eq 32 ]
then
	nodes=1
elif [ $1 -eq 64 ]
then
	nodes=2
elif [ $1 -eq 128 ]
then
	nodes=3
elif [ $1 -eq 235 ]
then
	nodes=5
elif [ $1 -eq 256 ]
then
	nodes=6
elif [ $1 -eq 512 ]
then
	nodes=11
else
	echo "$1 Not valid"
	exit 0
fi
echo > scal_out_file_"$1".txt
file=$(date +%s)
cp "/home/maths/strkss/massi/elm/Burgers.py" "/home/maths/strkss/massi/elm/Burgers_"$file".py"
for mdl in "elm" "para" "nngp"; do for dx in 128 1128; do
	sbatch <<EOT
#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 
module load GCCcore/11.3.0 Python/3.10.4

cd "/home/maths/strkss/massi/elm"
source venv/bin/activate

# Multiprocess application
srun python -u -m mpi4py.futures Burgers_"$file".py $1 $mdl $dx
#srun python -u -m mpi4py.futures Burgers_"$file".py $1 $mdl

exit 0

EOT
done
done
