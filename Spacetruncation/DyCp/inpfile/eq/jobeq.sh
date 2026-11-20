#!/bin/sh
#___INFO__MARK_BEGIN__
#  
#___INFO__MARK_END_ 
# 
#$ -S /bin/sh
#$ -N test_pyscf
#$ -j y
#$ -o ./
#$ -e ./
#$ -cwd
########## fat5 #############
#$ -q fat5.q

source ~/.bashrc
source /export/apps/Anaconda3_2019/settings.sh
conda activate p310

shopt -s extglob
export PYTHONPATH="/export/home/dengyf/LAMP_emb:$PYTHONPATH"
export PATH"=/export/home/dengyf/.local/bin:$PATH"


export TMPDIR=${PWD}
export TMP=${TMPDIR}
export TEMP=${TMPDIR}

date > stdout1
python dmet_cassiso.py >> stdout1
date >> stdout1
	
