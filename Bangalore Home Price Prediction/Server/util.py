import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns #in bangalore_house_pred_columns.json file, data_columns has locations. The json has to be loaded first to get the locations.
    global __locations

    with open("./Artifacts/bangalore_house_pred_columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[4:]  # first 4 columns are sqft,balcony, bath, bhk. We need locations which start from index 3

        global __model
        if __model is None:
            with open('./Artifacts/banglore_home_prices_model.pickle', 'rb') as f:
                __model = pickle.load(f)
        print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2) #2 here - rounding the price to 2 decimal places

def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location