#!/bin/bash
#SBATCH --job-name Wz_test
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=30:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/test.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=midway2-0690,midway2-0694


RUNNER=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/DELVERunner.py
RUNNER2=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/ThreeSDirRunner.py

#python -u $RUNNER --TrainRunner
#python -u $RUNNER --ClassifyRunner
#python -u $RUNNER --BinRunner
#python -u $RUNNER2 --ZPSetupRunner  --Nsamples 2
python -u $RUNNER2 --ZPUncertRunner --Nsamples 2
