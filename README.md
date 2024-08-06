# PhotovoltaicBasedCoudinessWeatherClassifier
In recent years, there has been an energy transition in
which fossil fuel-fired power plants are being replaced by renew-
able energy sources such as photovoltaics. While the influence of
factors such as location and method of installation or the sun’s po-
sition above the horizon is deterministic, the weather factor is rather
random. Weather data can be easily obtained from nearby weather
stations, but some weather phenomena are so dynamic that data ob-
tained in this way can become useless. When considering the prob-
lem of classifying weather conditions based on production data of
photovoltaic installations, we propose two new features extracted
from the production data that describe the overall level of sunshine
and the variability of sunlight. Three supervised learning methods
were used to classify the dataset: CNN, Random Forest classifier
and decision tree classifier. Reference values were obtained by gen-
erating classification results from the raw output and compared with
those generated from the two new features. The experiments showed
a statistically significant advantage for the models fitted to the data
extended with the new features.

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

`generate_paper_images.py` generates pdf file, that contains all images used in the article. The file is saved under the `cm/` directory.

# Datasets

Datasets are stored in `datasets` directory. `dataset_annotated.csv` is an extended file that contains experts annotations, while the `dataset.csv` is an original file. 

# Licence

The model is available under the Attribution 4.0 International (CC BY 4.0). You can use it 
in any purpose you want, however, the authors and source must be cited.

# More

[Dataset on kaggle](https://www.kaggle.com/datasets/kollosp/photovoltaic-dataset-with-weather-classification/data) 
