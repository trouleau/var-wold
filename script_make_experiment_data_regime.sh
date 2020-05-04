for n in {1000,1802,3246,5848,10536,18982,34200,61616,111009,200000}
do
    python script_make_job.py -e output/dataRegimes-n10/ -d 10 -n $n -g 10 -s 5
done
