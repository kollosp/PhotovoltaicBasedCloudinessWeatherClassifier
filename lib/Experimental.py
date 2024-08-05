from .Dataset import Dataset
from .Optimized import Optimized
from .Model import Model
from . import Time as tm 


def prepare_extended_dataset(
        model_x_bins=40,
        model_y_bins=30,
        feature_window_size=3,
        overall_window_size=tm._3H,
        variability_window_size=tm._3H,
        included_fields=None,
        limit_dataset=None):

    ts, production = Dataset.get(field=1)
    # _, model = Dataset.get(field=3)
    _, reference_weather = Dataset.get(field=7)

    if limit_dataset is not None:
        ts = ts[0:limit_dataset]
        production = production[0:limit_dataset]
        # model = model[0:limit_dataset]
        reference_weather = reference_weather[0:limit_dataset]

    latitude_degrees = 51
    longitude_degrees = 14

    m = Model(latitude_degrees, longitude_degrees, x_bins=model_x_bins, y_bins=model_y_bins)
    m.fit(ts, production)

    # predict data for selected timestamps
    expected_from_model = m.predict(ts)  # in-sample prediction - considered as expected production

    overall_cloudiness_index = Optimized.overall_cloudiness_index(production,
                                                                  expected_from_model,
                                                                  window_size=overall_window_size)
    variability_cloudiness_index = Optimized.variability_cloudiness_index(production, expected_from_model,
                                                                          window_size=variability_window_size)

    # fig, axis = Plotter.plot(ts, [production, variability_cloudiness_index, overall_cloudiness_index],
    #                          [(0, 4 * tm.DAY), (60 * DAY, 64 * tm.DAY),  (90 * tm.DAY, 94 * tm.DAY),  (120 * tm.DAY, 124 * tm.DAY)])

    data_ts = [production, expected_from_model, variability_cloudiness_index, overall_cloudiness_index]

    if included_fields is not None:
        data_ts = [dt for i,dt in enumerate(data_ts) if i in included_fields]

    return Dataset.timeseries_to_dataset(data_ts, window_size=feature_window_size),\
           Dataset.timeseries_to_dataset([reference_weather], window_size=1, skip=feature_window_size).astype(int) # classification

def prepare_dataset(feature_window_size:int=3, reference_weather_field=7, filename="weather_selection_annotated.csv"):
    # X and Y
    ts, production = Dataset.get(filename=filename)
    _, reference_weather = Dataset.get(field=reference_weather_field, filename=filename)
    return Dataset.timeseries_to_dataset([production], window_size=feature_window_size),\
           Dataset.timeseries_to_dataset([reference_weather], window_size=1, skip=feature_window_size).astype(int)  # classification
