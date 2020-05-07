exit 0; # Avoid execute these commands by mistake and destroying the universe

# ==============================================================================

# --- List non-empty output files
ls -lh ./*/stdout-* | awk '{if ($5 != 0) print $0}'

# --- List non-empty error files
ls -lh ./*/stderr-* | awk '{if ($5 != 0) print $0}'

# --- Number of output files
ls -l -1 ./*/output-*.json | wc -l

# --- Print same line from many files
{ for i in output/test-exp/g01-n000750/stdout-*; do echo "$i : $(sed '42q;d' "$i")"; done ;}

# --- Match output log files containing the keyword `Finished`
grep -rnw ./*/stdout-* -e Finished

# --- SSH tunnel for Jupyter server run remotely:
# `ssh -N -f -L localhost:<local-port>:localhost:<remote-port> <remote-host>`
ssh -N -f -L localhost:8899:localhost:2636 root@iccluster069.iccluster.epfl.ch

# --- Run condor script for many experiments at once
# search `in all subdirecories starting with 'dim' ` and execute the command
# 'command_submit' on the 'script.condor' file in each subdirectory
find ./dim* -maxdepth 0 | sort -R | xargs -n1 sh -c 'condor_submit $1/script.condor' sh

# --- Transfer output data from server to local (exclude log files)
rsync -rav --exclude='stdout-*' --exclude='stderr-*' root@iccluster069.iccluster.epfl.ch:<EXPERIMENT_FOLDER> .


# ==============================================================================

rsync -rav --exclude='stdout-*' --exclude='stderr-*' root@iccluster097.iccluster.epfl.ch:/root/workspace/var-wold/output/dimRegime-1/ .
