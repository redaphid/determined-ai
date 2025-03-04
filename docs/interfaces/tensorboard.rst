.. _tensorboards:

##############
 TensorBoards
##############

`TensorBoard <https://www.tensorflow.org/tensorboard>`__ is a widely used tool for visualizing and
inspecting deep learning models. Determined makes it easy to use TensorBoard to examine a single
experiment or to compare multiple experiments.

TensorBoard instances can be launched via the WebUI or the CLI. To launch TensorBoard instances from
the CLI, first :ref:`install the CLI <install-cli>` on your development machine.

*********************
 Analyze Experiments
*********************

To launch TensorBoard to analyze a single Determined experiment, use ``det tensorboard start
<experiment-id>``:

.. code::

   $ det tensorboard start 7
   Scheduling TensorBoard (rarely-cute-man) (id: aab49ba5-3357-4145-861c-7e6ff2d702c5)...
   TensorBoard (rarely-cute-man) was assigned to an agent...
   Scheduling tensorboard tensorboard (id: c68c9fc9-7eed-475b-a50f-fd78406d7c83)...
   TensorBoard is running at: http://localhost:8080/proxy/c68c9fc9-7eed-475b-a50f-fd78406d7c83/
   disconnecting websocket

The Determined master will schedule a TensorBoard instance in the cluster. The Determined CLI will
wait until the TensorBoard instance is running, and then it will open the TensorBoard web interface
in a local browser window.

You view information about scheduled and running TensorBoard instances by executing the following
command:

.. code::

   $ det tensorboard list
    Id                                   | Owner      | Description                         | State      | Experiment Id   | Trial Ids   | Exit Status
   --------------------------------------+------------+-------------------------------------+------------+-----------------+-------------+--------------
    aab49ba5-3357-4145-861c-7e6ff2d702c5 | determined | TensorBoard (rarely-cute-man)       | RUNNING    | 7               | N/A         | N/A

TensorBoard can also be used to analyze multiple experiments. To launch TensorBoard for multiple
experiments use ``det tensorboard start <experiment-id> <experiment-id> ...``.

.. note::

   Initially, TensorBoard may not contain metrics when the browser window opens. Data will be
   available after a trial workload is completed. TensorBoard pulls metrics from persistent storage.
   It may take up to 5 minutes for TensorBoard to receive data and render visualizations.

************************
 Customize TensorBoards
************************

Determined supports initializing TensorBoard with a YAML configuration file. For example, this
feature can be useful for running TensorBoard with a specific container image or for enabling access
to additional data with a bind-mount.

.. code:: yaml

   environment:
     image: determinedai/environments:cuda-11.3-pytorch-1.12-tf-2.8-gpu-0.19.12
   bind_mounts:
     - host_path: /my/agent/path
       container_path: /my/container/path
       read_only: true

Details of configuration settings can be found in the :ref:`command-notebook-configuration`.

To launch Tensorboard with a config file, use ``det tensorboard start <experiment-id>
--config-file=my_config.yaml``.

To view the configuration of a running Tensorboard instance, use ``det tensorboard config
<tensorboard_id>``.

*************************
 Analyze Specific Trials
*************************

Determined also supports using TensorBoard to analyze specific trials from one or more experiments.
This can be useful if an experiment has many trials but you would like to only compare a small
number of them. This capability can also be used to compare trials from *different* experiments.

To launch TensorBoard to analyze specific trials, use ``det tensorboard start --trial-ids <trial_id
1> <trial_id 2> ...``.

.. _data-in-tensorboard:

*********************
 Data in TensorBoard
*********************

In this section, we summarize how Determined captures data from TensorFlow models. For a more in
depth discussion of how TensorBoard visualizes data see the `TensorBoard documentation
<https://github.com/tensorflow/tensorboard/blob/master/README.md>`__.

TensorBoard visualizes data captured during model training and validation. Data is captured in
tfevent files by writing `TensorFlow summary operations
<https://www.tensorflow.org/api_docs/python/tf/summary>`__ to disk via a `tf.summary.FileWriter
<https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/summary/FileWriter>`__. We provide
support in each deep learning framework to write and upload metrics as tfevent files. See below for
details on how to configure Determined with TensorBoard for your desired framework.

FileWriters are configured to write log files, called tfevent files, to a directory known as the
``logdir``. TensorBoard watches this directory for changes and updates accordingly. The
Determined-supported ``logdir`` is ``/tmp/tensorboard``. All tfevent files written to
``/tmp/tensorboard`` in a trial are uploaded to persistent storage when a trial is configured with
Determined TensorBoard support.

**************************
 Determined Batch Metrics
**************************

At the end of every training workload, batch metrics are collected and stored in the database,
providing a granular view of model metrics over time. Batch metrics will appear in TensorBoard under
the Determined group. The x-axis of each plot corresponds to the batch number. For example, a point
at step 5 of the plot is the metric associated with the fifth batch seen.

**********************************
 Framework-specific Configuration
**********************************

The following examples demonstrate how to configure TensorBoard for each framework.

TensorFlow Keras
================

To add TensorBoard support for models that use :class:`~determined.keras.TFKerasTrial`, add a
:class:`determined.keras.callabacks.TensorBoard` callback to your trial class:

.. code:: python

   from determined.keras import TFKerasTrial
   from determined.keras.callbacks import TensorBoard


   class MyModel(TFKerasTrial):
       ...

       def keras_callbacks(self):
           return [TensorBoard()]

Estimator
=========

There is no configuration necessary for trials using :class:`~determined.estimator.EstimatorTrial`.
Unless configured otherwise, Estimators automatically log TensorBoard events to the ``model_dir``,
which Determined then moves to ``/tmp/tensorboard``.

PyTorch
=======

To add TensorBoard support for models that use the :doc:`PyTorch API
</training/apis-howto/api-pytorch-ug>`, use the ``writer`` field in an instance of the
:class:`~determined.tensorboard.metric_writers.pytorch.TorchWriter` class:

.. code:: python

   from determined.tensorboard.metric_writers.pytorch import TorchWriter


   class MyModel(PyTorchTrial):
       def __init__(self, context):
           ...
           self.logger = TorchWriter()

       def train_batch(self, batch, epoch_idx, batch_idx):
           self.logger.writer.add_scalar("my_metric", np.random.random(), batch_idx)

For a full-length example of using TensorBoard with PyTorch, see the :download:`mnist-GAN model
</examples/gan_mnist_pytorch.tgz>`.

**********************
 Lifecycle Management
**********************

Determined will automatically terminate idle TensorBoard instances. A TensorBoard instance is
considered idle if it is does not receive HTTP traffic (a TensorBoard that is still being viewed by
a web browser will not be considered idle). By default, idle TensorBoards will be terminated after 5
minutes; the timeout duration can be changed by editing ``tensorboard_timeout`` in the :ref:`master
config file <master-config-reference>`.

You can also terminate TensorBoard instances by hand using ``det tensorboard kill
<tensorboard-id>``:

.. code::

   $ det tensorboard kill aab49ba5-3357-4145-861c-7e6ff2d702c5

To open a web browser window connected to a previously launched TensorBoard instance, use ``det
tensorboard open``. To view the logs of an existing TensorBoard instance, use ``det tensorboard
logs``.

************************
 Implementation Details
************************

Determined schedules TensorBoard instances in containers that run on agent machines. The Determined
master will proxy HTTP requests to and from the TensorBoard container. TensorBoard instances are
hosted on agent machines but they do not occupy GPUs.

*****
 FAQ
*****

Can I log additional TensorBoard events beyond what Determined logs automatically?
==================================================================================

Yes; any additional TFEvent files that are written to ``/tmp/tensorboard`` inside a trial container
will be accessible via TensorBoard. For example, to log a custom TensorBoard event using PyTorch:

.. code:: python

   from torch.utils.tensorboard import SummaryWriter

   writer = SummaryWriter(log_dir="/tmp/tensorboard")
   writer.add_scalar("my_metric", np.random.random(), batch_idx)

For more details, as well as examples of how to do this with TF Estimator and TF Keras models, refer
to the :ref:`TensorBoard How-To Guide <data-in-tensorboard>`.

Can I use TensorBoard with PyTorch?
===================================

Yes! For an example of this check out the :download:`mnist-GAN </examples/gan_mnist_pytorch.tgz>`
example. This model uses the :class:`~determined.tensorboard.metric_writers.pytorch.TorchWriter`
class which automatically configures the location for writing TensorBoards. Users can also directly
use ``torch.utils.tensorboard.SummaryWriter`` as shown in the snippet above.
