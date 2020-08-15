import argparse
import json
import os
import pickle
import sys
import pandas as pd
import numpy as np
import glob
import tensorflow as tf

from utils import review_to_words, convert_and_pad



def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    print("calling input_handler")
    print("data:", data)
    # print("dir(data):", dir(data))
    # print("data.read():", data.read().decode('utf-8'))
    print("context:", context)
    
    # Load the saved word_dict.
    # print(os.system('pwd'))
    # print(os.system('ls -Rla ./'))
    # print(os.system('ls -Rla /opt/ml/model'))
    word_dict_path = os.path.join('/opt/ml/model', 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)
        
    print('word_dict:', word_dict)
    
    if context.request_content_type == 'application/json':
        d = data.read().decode('utf-8')
        
        print('Decoded data:', d)
        
        data_X, data_len = convert_and_pad(word_dict, review_to_words(d))
        data_X = np.array(data_X)[np.newaxis]
        data_X_json = json.dumps({
            'instances': data_X.tolist()
        })
        
        print('data_X_json:', data_X_json)
        
        return data_X_json

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))

def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    print("calling output_handler")
    print("data:", data)
    # print("dir(data):", dir(data))
    print("data.content:", data.content)
    # print("data.raw:", data.raw)
    print("data.text:", data.text)
    print("context:", context)
    
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type