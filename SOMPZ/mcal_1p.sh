#!/bin/bash
#SBATCH --job-name mcal_1p_SOM
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=60:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/log_%x
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=midway2-0694


BASE=/home/dhayaa/Desktop/DECADE/
RUNNER=CosmicShearPhotoZ.SOMPZ.DELVERunner
RUNNER2=CosmicShearPhotoZ.SOMPZ.ThreeSDirRunner

PYTHONPATH=$BASE python -m $RUNNER --AllMcalRunner --MCAL_TYPE 1p --njobs 15