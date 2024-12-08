
"""
This module contains functions for transforming restaurant reviews data.
"""

import tensorflow as tf

LABEL_KEY_NEW = "label"
FEATURE_KEY_NEW = "tweet"

def transformed_name(key):
    """
    Transform the given key.

    Args:
        key (str): Input key to transform.

    Returns:
        str: Transformed key.
    """
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input data.

    Args:
        inputs (dict): Input data dictionary containing 'label' and 'tweet' keys.

    Returns:
        dict: Transformed output data dictionary.
    """
    outputs = {}
    print(inputs[FEATURE_KEY_NEW])
    outputs[transformed_name(LABEL_KEY_NEW)] = tf.cast(inputs[LABEL_KEY_NEW], tf.int64)
    outputs[transformed_name(FEATURE_KEY_NEW)] = tf.strings.lower(inputs[FEATURE_KEY_NEW])
    return outputs
