#!/bin/bash


echo "Starting job: model"
sbatch sbatch/model.sbatch

#echo "Starting job: channel"
#sbatch sbatch/channel.sbatch

echo "Starting job: normalization"
sbatch sbatch/normalization.sbatch

echo "Starting job: diameter"
sbatch sbatch/diameter.sbatch

echo "Starting job: cellprob"
sbatch sbatch/cellprob.sbatch

echo "Starting job: min_size"
sbatch sbatch/min_size.sbatch

echo "Starting job: stitch"
sbatch sbatch/stitch.sbatch

echo "Starting job: tile"
sbatch sbatch/tile.sbatch

echo "Starting job: smoothing"
sbatch sbatch/smoothing.sbatch
