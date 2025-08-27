#!/bin/bash
#
#SBATCH --job-name=generate_embeddings                      # Job name
#SBATCH --output=logs/embedding_generation_run.%j.out       # Standard output (%j expands to jobId)
#SBATCH --error=logs/embedding_generation_run.%j.err        # Standard error
#SBATCH --partition=shared-gpu                              # not standard otherwise no permissions
#SBATCH --ntasks=1                                          # Run a single task
#SBATCH --cpus-per-task=1                                   # Number of CPU cores per task
#SBATCH --mem=64G                                           # Total memory
#SBATCH --gres=gpu:1                                        # Request one GPU
#SBATCH --exclude=gpu01.exbio.wzw.tum.de,                   # Exclude specific GPU node
#SBATCH --time=4-00:00:00                                   # Time limit hh:mm:ss
#SBATCH --mail-type=END,FAIL                                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=antspa@gmx.de                           # Where to send mail

# Run the Python script
python generate_embeddings.py