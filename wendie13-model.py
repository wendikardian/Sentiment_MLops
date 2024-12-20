# -*- coding: utf-8 -*-
"""SentimentDeployMLOPS.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1V7wWCLPyeQ6J-DtQOj6DBBFN6JA3o9DF

# Download Dataset dari kaggle
Sumber dataset : https://www.kaggle.com/datasets/dineshpiyasamara/sentiment-analysis-dataset
"""

!pip install tfx
!pip install autopep8
!pip install pylint

!pip install kaggle

# Create the .kaggle directory
!mkdir -p ~/.kaggle

# Move kaggle.json to the .kaggle directory
!mv kaggle.json ~/.kaggle/

# Set permissions for the kaggle.json file
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d dineshpiyasamara/sentiment-analysis-dataset

!mkdir data
!unzip sentiment-analysis-dataset.zip -d data
!ls data

!mkdir modules

"""# Menjalankan sistem Machine Learning"""

COMPONENTS_FILE = "modules/components.py"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TRAINER_MODULE_FILE =  "modules/trainer.py"
PIPELINES = "local_pipeline.py"
TUNER_MODULE_FILE = "modules/tuner.py"

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TUNER_MODULE_FILE}
# import keras_tuner as kt
# import tensorflow as tf
# import tensorflow_transform as tft
# from typing import NamedTuple, Dict, Text, Any
# from keras_tuner.engine import base_tuner
# from tensorflow.keras import layers
# from tfx.components.trainer.fn_args_utils import FnArgs
# from tensorflow.keras.callbacks import EarlyStopping
# 
# LABEL_KEY = "label"
# FEATURE_KEY = "tweet"
# NUM_EPOCHS = 2
# 
# TunerFnResult = NamedTuple("TunerFnResult", [
#     ("tuner", base_tuner.BaseTuner),
#     ("fit_kwargs", Dict[Text, Any]),
# ])
# 
# early_stopping_callback = EarlyStopping(
# monitor='binary_accuracy',  # Monitoring the binary accuracy
# patience=3,  # Number of epochs with no improvement after which training will be stopped
# min_delta=0.01,  # Minimum change in the monitored quantity to qualify as an improvement
# mode='max',  # Mode should be 'max' since we want to maximize the accuracy
# baseline=0.85  # Stop training once the accuracy reaches 85%
# )
# 
# def transformed_name(key):
#     return f"{key}_xf"
# 
# 
# def gzip_reader_fn(filenames):
#     return tf.data.TFRecordDataset(filenames, compression_type="GZIP")
# 
# 
# def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
#     transform_feature_spec = (
#         tf_transform_output.transformed_feature_spec().copy()
#     )
# 
#     dataset = tf.data.experimental.make_batched_features_dataset(
#         file_pattern=file_pattern,
#         batch_size=batch_size,
#         features=transform_feature_spec,
#         reader=gzip_reader_fn,
#         num_epochs=num_epochs,
#         label_key=transformed_name(LABEL_KEY),
#     )
# 
#     return dataset
# 
# 
# def model_builder(hp, vectorizer_layer):
#     num_hidden_layers = hp.Choice(
#         "num_hidden_layers", values=[1, 2]
#     )
#     embed_dims = hp.Int(
#         "embed_dims", min_value=16, max_value=128, step=32
#     )
#     lstm_units= hp.Int(
#         "lstm_units", min_value=32, max_value=128, step=32
#     )
#     dense_units = hp.Int(
#         "dense_units", min_value=32, max_value=256, step=32
#     )
#     dropout_rate = hp.Float(
#         "dropout_rate", min_value=0.1, max_value=0.5, step=0.1
#     )
#     learning_rate = hp.Choice(
#         "learning_rate", values=[1e-2, 1e-3, 1e-4]
#     )
# 
#     inputs = tf.keras.Input(
#         shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
#     )
# 
#     x = vectorizer_layer(inputs)
#     x = layers.Embedding(input_dim=5000, output_dim=embed_dims)(x)
#     x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
# 
#     for _ in range(num_hidden_layers):
#         x = layers.Dense(dense_units, activation=tf.nn.relu)(x)
#         x = layers.Dropout(dropout_rate)(x)
# 
#     outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)
# 
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#         loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#         metrics=["binary_accuracy"],
#     )
# 
#     return model
# 
# 
# def tuner_fn(fn_args: FnArgs):
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
# 
#     train_set = input_fn(
#         fn_args.train_files[0], tf_transform_output, NUM_EPOCHS
#     )
#     eval_set = input_fn(
#         fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS
#     )
# 
#     vectorizer_dataset = train_set.map(
#         lambda f, l: f[transformed_name(FEATURE_KEY)]
#     )
# 
#     vectorizer_layer = layers.TextVectorization(
#         max_tokens=5000,
#         output_mode="int",
#         output_sequence_length=500,
#     )
#     vectorizer_layer.adapt(vectorizer_dataset)
# 
#     def wrapped_model_builder(hp):
#         # Wrap the `model_builder` to include `vectorizer_layer`
#         return model_builder(hp, vectorizer_layer)
# 
#     hp = kt.HyperParameters()
#     hp.Choice('learning_rate', [1e-1, 1e-3])
#     hp.Int('num_layers', 1, 5)
# 
#     tuner = kt.RandomSearch(
#         wrapped_model_builder,  # Use the wrapper function
#         max_trials=NUM_EPOCHS,
#         hyperparameters=hp,
#         allow_new_entries=True,
#         objective='val_accuracy',
#         directory=fn_args.working_dir,
#         project_name='test'
#     )
#     return TunerFnResult(
#         tuner=tuner,
#         fit_kwargs={
#             "callbacks": [early_stopping_callback],
#             "x": train_set,
#             "validation_data": eval_set,
#             "steps_per_epoch": fn_args.train_steps,
#             "validation_steps": fn_args.eval_steps,
#         },
#     )

# Commented out IPython magic to ensure Python compatibility.
# 
# %%writefile {COMPONENTS_FILE}
# import os
# import tensorflow_model_analysis as tfma
# 
# from tfx.components import (
#     CsvExampleGen,
#     StatisticsGen,
#     SchemaGen,
#     ExampleValidator,
#     Transform,
#     Trainer,
#     Evaluator,
#     Pusher,
#     Tuner
# )
# from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
# from tfx.types import Channel
# from tfx.dsl.components.common.resolver import Resolver
# from tfx.types.standard_artifacts import Model, ModelBlessing
# from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
#     LatestBlessedModelStrategy
# )
# 
# 
# def init_components(
#     data_dir,
#     transform_module,
#     training_module,
#     training_steps,
#     eval_steps,
#     serving_model_dir,
# ):
#     """
#     Initializes and configures TFX components for building a pipeline
#     to train and deploy a machine learning model.
# 
#     Args:
#         data_dir (str): Path to the directory containing input data for the pipeline.
#         transform_module (str): Path to the Python module implementing data transformation logic.
#         training_module (str): Path to the Python module defining the model training logic.
#         training_steps (int): Number of steps to execute during model training.
#         eval_steps (int): Number of steps to execute during model evaluation.
#         serving_model_dir (str): Path to the directory where the trained model will be exported for deployment.
#     """
# 
#     output = example_gen_pb2.Output(
#         split_config=example_gen_pb2.SplitConfig(splits=[
#             example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
#             example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
#         ])
#     )
# 
#     example_gen = CsvExampleGen(
#         input_base=data_dir,
#         output_config=output
#     )
# 
#     statistics_gen = StatisticsGen(
#         examples=example_gen.outputs['examples']
#     )
# 
#     schema_gen = SchemaGen(
#         statistics=statistics_gen.outputs["statistics"]
#     )
# 
#     example_validator = ExampleValidator(
#         statistics=statistics_gen.outputs['statistics'],
#         schema=schema_gen.outputs['schema']
#     )
# 
#     transform = Transform(
#         examples=example_gen.outputs['examples'],
#         schema=schema_gen.outputs['schema'],
#         module_file=os.path.abspath(transform_module)
#     )
# 
#     trainer = Trainer(
#         module_file=os.path.abspath(training_module),
#         examples=transform.outputs['transformed_examples'],
#         transform_graph=transform.outputs['transform_graph'],
#         schema=schema_gen.outputs['schema'],
#         train_args=trainer_pb2.TrainArgs(
#             splits=['train'],
#             num_steps=training_steps
#         ),
#         eval_args=trainer_pb2.EvalArgs(
#             splits=['eval'],
#             num_steps=eval_steps
#         )
#     )
# 
#     model_resolver = Resolver(
#         strategy_class=LatestBlessedModelStrategy,
#         model=Channel(type=Model),
#         model_blessing=Channel(type=ModelBlessing)
#     ).with_id('Latest_blessed_model_resolver')
# 
#     eval_config = tfma.EvalConfig(
#         model_specs=[tfma.ModelSpec(label_key='label')],
#         slicing_specs=[tfma.SlicingSpec()],
#         metrics_specs=[
#             tfma.MetricsSpec(metrics=[
#                 tfma.MetricConfig(class_name='ExampleCount'),
#                 tfma.MetricConfig(class_name='AUC'),
#                 tfma.MetricConfig(class_name='FalsePositives'),
#                 tfma.MetricConfig(class_name='TruePositives'),
#                 tfma.MetricConfig(class_name='FalseNegatives'),
#                 tfma.MetricConfig(class_name='TrueNegatives'),
#                 tfma.MetricConfig(class_name='BinaryAccuracy',
#                     threshold=tfma.MetricThreshold(
#                         value_threshold=tfma.GenericValueThreshold(
#                             lower_bound={'value': 0.5}
#                         ),
#                         change_threshold=tfma.GenericChangeThreshold(
#                             direction=tfma.MetricDirection.HIGHER_IS_BETTER,
#                             absolute={'value': 0.0001}
#                         )
#                     )
#                 )
#             ])
#         ]
#     )
# 
#     evaluator = Evaluator(
#         examples=example_gen.outputs['examples'],
#         model=trainer.outputs['model'],
#         baseline_model=model_resolver.outputs['model'],
#         eval_config=eval_config
#     )
# 
#     pusher = Pusher(
#         model=trainer.outputs["model"],
#         model_blessing=evaluator.outputs["blessing"],
#         push_destination=pusher_pb2.PushDestination(
#             filesystem=pusher_pb2.PushDestination.Filesystem(
#                 base_directory=serving_model_dir
#             )
#         ),
#     )
#     tuner = Tuner(
#         module_file=os.path.abspath("modules/tuner.py"),
#         examples=transform.outputs["transformed_examples"],
#         transform_graph=transform.outputs["transform_graph"],
#         schema=schema_gen.outputs["schema"],
#         train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=training_steps),
#         eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps),
#     )
# 
# 
#     components = (
#         example_gen,
#         statistics_gen,
#         schema_gen,
#         example_validator,
#         transform,
#         tuner,
#         trainer,
#         model_resolver,
#         evaluator,
#         pusher
#     )
# 
#     return components

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TRAINER_MODULE_FILE}
# 
# """
# This module contains functions for training a sentiment analysis model using TensorFlow and TensorFlow Transform.
# """
# 
# import os
# import tensorflow as tf
# import tensorflow_transform as tft
# from tensorflow.keras import layers
# from tfx.components.trainer.fn_args_utils import FnArgs
# 
# # Define constants
# LABEL_KEY = "label"
# FEATURE_KEY = "tweet"
# EMBEDDING_DIM = 32
# 
# # Function to rename transformed features
# def transformed_name(key):
#     """
#     Transform the given key.
# 
#     Args:
#         key (str): Input key to transform.
# 
#     Returns:
#         str: Transformed key.
#     """
#     return key + "_xf"
# 
# # Function to read data from compressed TFRecord files
# def gzip_reader_fn(filenames):
#     return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
# 
# # Input function to create transformed features and batch data
# def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
#     """
#     Create input function for training data.
# 
#     Args:
#         file_pattern (str): File pattern for input data.
#         tf_transform_output (tensorflow_transform.TFTransformOutput): TensorFlow Transform output.
#         num_epochs (int): Number of epochs.
#         batch_size (int): Batch size.
# 
#     Returns:
#         tf.data.Dataset: Input dataset.
#     """
#     transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
#     dataset = tf.data.experimental.make_batched_features_dataset(
#         file_pattern=file_pattern,
#         batch_size=batch_size,
#         features=transform_feature_spec,
#         reader=gzip_reader_fn,
#         num_epochs=num_epochs,
#         label_key=transformed_name(LABEL_KEY)
#     )
#     return dataset
# 
# # Text vectorization layer for tokenization and data standardization
# vectorize_layer = layers.TextVectorization(
#     standardize="lower_and_strip_punctuation",
#     max_tokens=10000,
#     output_mode='int',
#     output_sequence_length=100
# )
# 
# # Function to build the machine learning model
# def model_builder():
#     """
#     Build the machine learning model.
# 
#     Returns:
#         tf.keras.Model: Compiled Keras model.
#     """
#     inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
#     reshaped_input = tf.reshape(inputs, [-1])
#     x = vectorize_layer(reshaped_input)
#     x = layers.Embedding(10000, EMBEDDING_DIM, name="embedding")(x)
#     x = layers.GlobalAveragePooling1D()(x)
#     x = layers.Dense(64, activation="relu")(x)
#     x = layers.Dense(32, activation="relu")(x)
#     outputs = layers.Dense(1, activation="sigmoid")(x)
# 
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=tf.keras.optimizers.Adam(0.01),
#         metrics=[tf.keras.metrics.BinaryAccuracy()]
#     )
# 
#     model.summary()
#     return model
# 
# # Function to preprocess raw request data for deployment
# def _get_serve_tf_examples_fn(model, tf_transform_output):
#     """
#     Get serving function for TensorFlow Serving.
# 
#     Args:
#         model (tf.keras.Model): Trained Keras model.
#         tf_transform_output (tensorflow_transform.TFTransformOutput): TensorFlow Transform output.
# 
#     Returns:
#         Callable: Serve function for TensorFlow Serving.
#     """
#     model.tft_layer = tf_transform_output.transform_features_layer()
# 
#     @tf.function
#     def serve_tf_examples_fn(serialized_tf_examples):
#         feature_spec = tf_transform_output.raw_feature_spec()
#         feature_spec.pop(LABEL_KEY)
#         parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
#         transformed_features = model.tft_layer(parsed_features)
#         return model(transformed_features)
# 
#     return serve_tf_examples_fn
# 
# # Function to run the training process
# def run_fn(fn_args: FnArgs) -> None:
#     """
#     Run the training process.
# 
#     Args:
#         fn_args (FnArgs): Function arguments.
#     """
#     log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
# 
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=log_dir, update_freq='batch'
#     )
# 
#     es = tf.keras.callbacks.EarlyStopping(
#         monitor='val_binary_accuracy', mode='max', verbose=1, patience=10
#     )
#     mc = tf.keras.callbacks.ModelCheckpoint(
#         fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True
#     )
# 
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
# 
#     train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
#     val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
#     vectorize_layer.adapt(
#         [j[0].numpy()[0] for j in [i[0][transformed_name(FEATURE_KEY)] for i in list(train_set)]]
#     )
# 
#     model = model_builder()
# 
#     model.fit(
#         x=train_set,
#         validation_data=val_set,
#         callbacks=[tensorboard_callback, es, mc],
#         steps_per_epoch=1000,
#         validation_steps=1000,
#         epochs=10
#     )
# 
#     signatures = {
#         'serving_default':
#         _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
#             tf.TensorSpec(
#                 shape=[None],
#                 dtype=tf.string,
#                 name='examples'
#             )
#         )
#     }
#     model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TRANSFORM_MODULE_FILE}
# 
# """
# Process of transforming sentiment data.
# """
# 
# import tensorflow as tf
# 
# LABEL_KEY_NEW = "label"
# FEATURE_KEY_NEW = "tweet"
# 
# def transformed_name(key):
#     """
#     Transform the given key.
# 
#     Args:
#         key (str): Input key to transform.
# 
#     Returns:
#         str: Transformed key.
#     """
#     return key + "_xf"
# 
# def preprocessing_fn(inputs):
#     """
#     Preprocess input data.
# 
#     Args:
#         inputs (dict): Input data dictionary containing 'label' and 'tweet' keys.
# 
#     Returns:
#         dict: Transformed output data dictionary.
#     """
#     outputs = {}
#     print(inputs[FEATURE_KEY_NEW])
#     outputs[transformed_name(LABEL_KEY_NEW)] = tf.cast(inputs[LABEL_KEY_NEW], tf.int64)
#     outputs[transformed_name(FEATURE_KEY_NEW)] = tf.strings.lower(inputs[FEATURE_KEY_NEW])
#     return outputs

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {PIPELINES}
# import os
# from typing import Text
# 
# from absl import logging
# from tfx.orchestration import metadata, pipeline
# from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
# 
# PIPELINE_NAME = 'sentiment-pipeline'
# 
# DATA_ROOT = 'data'
# TRANSFORM_MODULE_FILE = 'modules/transform.py'
# TRAINER_MODULE_FILE = 'modules/trainer.py'
# TUNER_MODULE_FILE = 'modules/tuner.py'
# 
# OUTPUT_BASE = 'output'
# serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
# pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
# metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')
# 
# 
# def init_local_pipeline(
#     components, pipeline_root: Text
# ) -> pipeline.Pipeline:
#     """
#     Initialize a local TFX pipeline.
# 
#     Args:
#         components: A dictionary of TFX components to be included in the pipeline.
#         pipeline_root: Root directory for pipeline output artifacts.
# 
#     Returns:
#         A TFX pipeline.
#     """
#     logging.info(f'Pipeline root set to: {pipeline_root}')
# 
#     return pipeline.Pipeline(
#         pipeline_name=PIPELINE_NAME,
#         pipeline_root=pipeline_root,
#         components=components,
#         enable_cache=True,
#         metadata_connection_config=metadata.sqlite_metadata_connection_config(
#             metadata_path
#         ),
#     )
# 
# 
# if __name__ == '__main__':
#     logging.set_verbosity(logging.INFO)
# 
#     from modules.components import init_components
#     components = init_components(
#         DATA_ROOT,
#         training_module=TRAINER_MODULE_FILE,
#         transform_module=TRANSFORM_MODULE_FILE,
#         training_steps=100,
#         eval_steps=50,
#         serving_model_dir=serving_model_dir,
#     )
# 
#     pipeline = init_local_pipeline(components, pipeline_root)
#     BeamDagRunner().run(pipeline=pipeline)

!python local_pipeline.py

!zip -r /content/output.zip /content/output

!autopep8 --in-place --aggressive --aggressive modules/components.py ./local_pipeline.py modules/transform.py modules/trainer.py modules/tuner.py

!pylint modules/components.py ./local_pipeline.py modules/transform.py modules/trainer.py modules/tuner.py

