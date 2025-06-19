#!/bin/bash
#SBATCH --job-name=your_executable
#SBATCH --output=out.log
#SBATCH --error=err.log
#SBATCH --time=00:02:00
#SBATCH --partition=partition_name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=your_group_name

module load cuda/11.8.0

./your_executables
