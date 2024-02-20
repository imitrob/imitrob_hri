
# Merging language and gestures

Run tests with:
```
python tester.py
```
Generate data with:
```
cd ../data
python data_creator.py
```

# Do everything in once

Generate datasets and run merging on all after they are finished
```
# term 1
python data/data_creator2.py all
# term 2
sleep 1000; python merging_modalities/tester_on_data2.py all
```