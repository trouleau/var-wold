for n in {499,753,1134,1709,2576,3881,5848,8810,13274,20000}
do
    python script_make_job.py -e output/dataRegimes-n10-3/ -d 10 -n $n -g 5 -s 4 --seed 545693892
done
