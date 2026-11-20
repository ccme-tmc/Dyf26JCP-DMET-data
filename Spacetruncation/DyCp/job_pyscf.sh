#!/bin/bash
#SBATCH -o stdout
#SBATCH --partition=cps6
#SBATCH -J DyCBrtp11
#SBATCH --get-user-env
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mail-type=end
#SBATCH --mail-user=sunhuaiyang@pku.edu.cn
#SBATCH --time=100:00:00

#bin="/THL6/home/jiangh/program/vasp.5.4.4/bin/vasp_std"
module load ase/3.22.1-gcc8.5.0

date > rtout2
python Analyze_CFPs_all.py >> rtout2
date >> rtout2

date > rtout
python Analyze_CFPs.py >> rtout
date >> rtout

