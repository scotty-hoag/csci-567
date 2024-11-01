@echo off
python3 makedata.py -makeinfo 0.90
python3 makedata.py -train
python3 makedata.py -test