import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow.keras.callbacks import EarlyStopping

LABEL_KEY = "label"
FEATURE_KEY = "tweet"
NUM_EPOCHS = 2

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

early_stopping_callback = EarlyStopping(
    monitor='binary_accuracy',  # Monitoring the binary accuracy
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    min_delta=0.01,  # Minimum change in the monitored quantity to qualify as an improvement
    mode='max',  # Mode should be 'max' since we want to maximize the accuracy
    baseline=0.85  # Stop training once the accuracy reaches 85%
)


def transformed_name(key):
    return f"{key}_xf"


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def model_builder(hp, vectorizer_layer):
    num_hidden_layers = hp.Choice(
        "num_hidden_layers", values=[1, 2]
    )
    embed_dims = hp.Int(
        "embed_dims", min_value=16, max_value=128, step=32
    )
    lstm_units = hp.Int(
        "lstm_units", min_value=32, max_value=128, step=32
    )
    dense_units = hp.Int(
        "dense_units", min_value=32, max_value=256, step=32
    )
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.5, step=0.1
    )
    learning_rate = hp.Choice(
        "learning_rate", values=[1e-2, 1e-3, 1e-4]
    )

    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
    )

    x = vectorizer_layer(inputs)
    x = layers.Embedding(input_dim=5000, output_dim=embed_dims)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)

    for _ in range(num_hidden_layers):
        x = layers.Dense(dense_units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["binary_accuracy"],
    )

    return model


def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(
        fn_args.train_files[0], tf_transform_output, NUM_EPOCHS
    )
    eval_set = input_fn(
        fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS
    )

    vectorizer_dataset = train_set.map(
        lambda f, l: f[transformed_name(FEATURE_KEY)]
    )

    vectorizer_layer = layers.TextVectorization(
        max_tokens=5000,
        output_mode="int",
        output_sequence_length=500,
    )
    vectorizer_layer.adapt(vectorizer_dataset)

    def wrapped_model_builder(hp):
        # Wrap the `model_builder` to include `vectorizer_layer`
        return model_builder(hp, vectorizer_layer)

    hp = kt.HyperParameters()
    hp.Choice('learning_rate', [1e-1, 1e-3])
    hp.Int('num_layers', 1, 5)

    tuner = kt.RandomSearch(
        wrapped_model_builder,  # Use the wrapper function
        max_trials=NUM_EPOCHS,
        hyperparameters=hp,
        allow_new_entries=True,
        objective='val_accuracy',
        directory=fn_args.working_dir,
        project_name='test'
    )
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping_callback],
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
