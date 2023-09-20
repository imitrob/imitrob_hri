# imitrob-hri

gesture, nlp and dialogue ros2 packages for common imitrob setup, jointly working with crow-base modules

## Merge modalities feature

Run tester: `python tester.py` (in `imitrob-hri/merging_modalities`)

### Install additional: (might be added to requirements or conda yml file)

```
pip install owlready
pip install nltk
pip install gTTS
pip install playsound
pip install SpeechRecognition
pip install rdflib
pip install knowl
pip install python-dotenv
```

### Usage

Dataset generation:
- Generate all datasets (~5min): `python data_creator.py all`
- Generate single dataset: `python data_creator.py cX_nY_DZ` (e.g. `python data_creator.py c1_n2_D3`)
    - where X is configuration choice int from 1 to 3
    - Y is noise levels int from 1 to 3  
    - Z is dataset number from 1 to 5
- This creates new datasets into `data/saves` folder

Dataset tests:
- Test on all configurations & all merge functions: `python tester_on_data.py all`
- Test on all configurations and single merge function: `python tester_on_data.py all MERGEFUN`, where `MERGEFUN` is string from list ['mul', 'add_2', 'entropy', 'entropy_add_2', 'baseline']
- Test on single configuration dataset `python tester_on_data.py cX_nY_DZ MERGEFUN MODEL`, where `MODEL` is picked from int: 0 (M1), 1 (M2), 2 (M3) (e.g. `python tester_on_data.py c1_n2_D3 mul 2`)
TODO: Better would be assign M1 string

Further tests:
- Check each samples in dataset `python tester_on_single_data.py`
- Check consistency (results time invariant): `python check_consistency_on_data.py all MERGEFUN MODEL`
`python check_consistency_on_data.py cX_nY_DZ MERGEFUN MODEL`

- Check dataset (e.g. how many objects with given properties are there, ...) `python data_checker.py cX_nX_DX`
- Experiments generation plots: (generation from npy array results folder) `python results_comparer.py`
- Histogram for configurations cX_nY_DZ across X axis `python results_plotter.py MERGEFUN`
    - Commited quick fix on line 74 recently (added `results/`)
