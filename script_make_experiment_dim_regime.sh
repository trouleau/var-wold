for d in {5,10,15,20,25,30,35,40,45,50}
do
    n=$((10000*d))
    python script_make_job.py -e output/dimRegime-5/ -d $d -n $n -g 5 -s 4
done
