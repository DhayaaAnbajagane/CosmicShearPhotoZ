#!/bin/bash
#SBATCH --job-name SOMPZ_RUN
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=60:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/log_run
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=midway2-0696,midway2-0694


RUNNER=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/DELVERunner.py
RUNNER2=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/ThreeSDirRunner.py

#python -u $RUNNER --TrainRunner --WIDE #--DEEP
#python -u $RUNNER --ClassifyRunner --BALROG --WIDE #--DEEP
#python -u $RUNNER --BinRunner

#python -u $RUNNER2 --ZPOffsetRunner --Nsamples 100
#python -u $RUNNER2 --ZPUncertRunner --Nsamples 100

#python -u $RUNNER2 --ThreeSDirRedbiasRunner --Nsamples 100
python -u $RUNNER2 --FinalRunner --Nsamples 100

#python -u $RUNNER --AllMcalRunner
