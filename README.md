# PhotovoltaicBasedCoudinessWeatherClassifier
 Photovoltaic based Coudiness Weather Classifier



# Project structure
```
.
├── cm
│   └── paper_images.pdf
├── datasets
│   ├── dataset_annotated.csv
│   └── dataset.csv
├── extended.pkl
├── generate_paper_images.py
├── lib
│   ├── Dataset.py
│   ├── Experimental.py
│   ├── __init__.py
│   ├── Model.py
│   ├── Optimized.py
│   ├── Plotter.py
│   ├── Time.py
│   ├── TimeSeriesAnnotation.py
│   └── Utils.py
├── perform_experiments.py
├── Photovoltaic based Coudiness Weather Classifier.pdf
├── raw.pkl
├── README.md
└── requirements.txt
```

Datasets are stored in `datasets` directory. `dataset_annotated.csv` is an extended file that contains experts annotations, while the `dataset.csv` is an original file. 

# Install & Run
Create python virtual environement and install libs.
```python
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install -r requirements.txt
```
There are two scripts provided in the project root `perform_experiments.py` and `generate_paper_images.py`. 
```python
(.venv) $ python perform_experiments.py
(.venv) $ python generate_paper_images.py
```
