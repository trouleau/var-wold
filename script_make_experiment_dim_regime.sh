for d in {5,10,15,20,25,30,35,40,45,50}
do
    n=$((7000*d))
    python script_make_job.py -e output/dimRegime-3/ -d $d -n $n -g 5 -s 3
done
