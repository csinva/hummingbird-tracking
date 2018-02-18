#!/usr/bin/env bash


############################### parameters to change ###############################
csv_dir="out_good"
label_file="../data/top/labels/fastec/good.csv" # optional, if no labels won't plot labels
####################################################################################


python3 view_thetas.py --csv_dir $csv_dir --label_file $label_file
