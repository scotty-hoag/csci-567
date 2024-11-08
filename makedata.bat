@echo off
python3 makedata.py -makeog
python3 makedata.py -makeinfo 0.90 -seed 13
python3 makedata.py -train
python3 makedata.py -test

python code/load_data.py -c -o -train
python code/load_data.py -o -train
python code/load_data.py -c -o -test
python code/load_data.py -o -test