#!/bin/bash
#SBATCH --job-name=RLTest1
#SBATCH --account=def-edelage
#SBATCH --mail-user=saeed.marzban@hec.ca
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=10
#SBATCH --output=output1.txt
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:p100:1
module load python/3.6
source ~/saeedenv36/bin/activate
srun python wavecorr_run.py --dataset_name can --network_model waveCorr