for d in {5,10,15,20,25,30,35,40,45,50}
do
    n=$((5000*d))
    python script_make_job.py -e output/dimRegime-2/ -d $d -n $n -g 10 -s 5
done
