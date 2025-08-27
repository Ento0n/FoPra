#!/bin/bash
#
#SBATCH --job-name=test_sequence_identity_parallel                         # Job name
#SBATCH --output=logs/test_sequence_identity_parallel_run.%j.out         # Standard output (%j expands to jobId)
#SBATCH --error=logs/test_sequence_identity_parallel_run.%j.err          # Standard error
#SBATCH --partition=shared-cpu                               # not standard otherwise no permissions
#SBATCH --ntasks=1                                          # Run a single task
#SBATCH --cpus-per-task=10                                  # Number of CPU cores per task
#SBATCH --mem=128G                                           # Total memory
#SBATCH --time=4-00:00:00                                   # Time limit hh:mm:ss
#SBATCH --mail-type=END,FAIL                                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=antspa@gmx.de                           # Where to send mail

# Run the Python script
python test_sequence_identity_parallel.py --deleak_uniprot --deleak_cdhit --local