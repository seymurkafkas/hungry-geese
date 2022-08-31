import pickle
import zlib
import base64 as b64
import numpy as np
from tensorflow import keras


def serialize_and_compress(value, verbose=True):
    serialized_value = pickle.dumps(value)
    if verbose:
        print("Lenght of serialized object:", len(serialized_value))
    c_data = zlib.compress(serialized_value, 9)
    if verbose:
        print("Lenght of compressed and serialized object:", len(c_data))
    return b64.b64encode(c_data)


def decompress_and_deserialize(compressed_data):
    d_data_byte = b64.b64decode(compressed_data)
    data_byte = zlib.decompress(d_data_byte)
    value = pickle.loads(data_byte)
    return value


def serialize_model_into_string(modelFilePath, outputFilePath):
    model = keras.models.load_model(modelFilePath)  # Create model here
    model_weights = model.get_weights()
    string_res = serialize_and_compress(model_weights)
    f = open(outputFilePath, "w")
    f.write(string_res.decode("utf-8"))
    f.close()
