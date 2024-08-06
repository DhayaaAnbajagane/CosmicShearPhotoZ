#!/bin/bash
#SBATCH --job-name NZ_Summary
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=140:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/log_summary
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=midway2-0696,midway2-0694


BASE=/home/dhayaa/Desktop/DECADE/
RUNNER=CosmicShearPhotoZ.SOMPZ.DELVERunner
RUNNER2=CosmicShearPhotoZ.SOMPZ.ThreeSDirRunner

PYTHONPATH=$BASE python -m $RUNNER2 --Summarize

