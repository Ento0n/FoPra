#!/bin/bash
#
#SBATCH --job-name=generate_contact_maps                         # Job name
#SBATCH --output=logs/contact_map_generation_run.%j.out         # Standard output (%j expands to jobId)
#SBATCH --error=logs/contact_map_generation_run.%j.err          # Standard error
#SBATCH --partition=shared-cpu                               # not standard otherwise no permissions
#SBATCH --ntasks=1                                          # Run a single task
#SBATCH --cpus-per-task=1                                   # Number of CPU cores per task
#SBATCH --mem=64G                                           # Total memory
#SBATCH --time=4-00:00:00                                   # Time limit hh:mm:ss
#SBATCH --mail-type=END,FAIL                                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=antspa@gmx.de                           # Where to send mail

# Run the Python script
python create_contact_maps.py