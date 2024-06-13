#!/bin/bash
#SBATCH --job-name mcal_2m_SOM
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=60:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/log_%x
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=midway2-0694


RUNNER=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/DELVERunner.py
RUNNER2=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/ThreeSDirRunner.py

python -u $RUNNER --AllMcalRunner --MCAL_TYPE 2m --njobs 15
