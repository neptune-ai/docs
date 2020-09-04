:mod:`neptune_tensorboard.integration.tensorflow_integration`
=============================================================

.. py:module:: neptune_tensorboard.integration.tensorflow_integration


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptune_tensorboard.integration.tensorflow_integration.TensorflowIntegrator



Functions
~~~~~~~~~

.. autoapisummary::

   neptune_tensorboard.integration.tensorflow_integration.integrate_with_tensorflow
   neptune_tensorboard.integration.tensorflow_integration._integrate_with_tensorflow
   neptune_tensorboard.integration.tensorflow_integration._patch_tensorflow_1x
   neptune_tensorboard.integration.tensorflow_integration._patch_tensorflow_2x


.. data:: _integrated_with_tensorflow
   :annotation: = False

   

.. function:: integrate_with_tensorflow(experiment_getter, prefix=False)


.. py:class:: TensorflowIntegrator(prefix=False, experiment_getter=None)

   Bases: :class:`future.builtins.object`

   A magical object class that provides Python 2 compatibility methods::
       next
       __unicode__
       __nonzero__

   Subclasses of this class can merely define the Python 3 methods (__next__,
   __str__, and __bool__).

   .. method:: get_channel_name(self, writer, name)


   .. method:: add_summary(self, writer, summary, global_step=None)


   .. method:: add_value(self, x, value, writer)


   .. method:: send_numeric(self, tag, step, value, wall_time)


   .. method:: send_image(self, tag, step, encoded_image_string, wall_time)


   .. method:: send_text(self, tag, step, text, wall_time)


   .. staticmethod:: get_writer_name(log_dir)


   .. staticmethod:: _calculate_x_value(global_step)



.. function:: _integrate_with_tensorflow(experiment_getter, prefix=False)


.. function:: _patch_tensorflow_1x(tensorflow_integrator)


.. function:: _patch_tensorflow_2x(experiment_getter, prefix)


