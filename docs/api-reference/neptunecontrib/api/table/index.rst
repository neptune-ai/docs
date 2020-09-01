:mod:`neptunecontrib.api.table`
===============================

.. py:module:: neptunecontrib.api.table


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.api.table.log_table


.. function:: log_table(name, table, experiment=None)

   Logs pandas dataframe to neptune.

   Pandas dataframe is converted to an HTML table and logged to Neptune as an artifact with path tables/{name}.html

   :param name: | Name of the chart (without extension) that will be used as a part of artifact's destination.
   :type name: :obj:`str`
   :param table: | DataFrame table
   :type table: :obj:`pandas.Dataframe`
   :param experiment:
                      | For advanced users only. Pass Neptune
                        `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
                        object if you want to control to which experiment data is logged.
                      | If ``None``, log to currently active, and most recent experiment.
   :type experiment: :obj:`neptune.experiments.Experiment`, optional, default is ``None``

   .. rubric:: Examples

   Start an experiment::

       import neptune

       neptune.init(api_token='ANONYMOUS',
                    project_qualified_name='shared/showroom')
       neptune.create_experiment(name='experiment_with_tables')

   Create or load dataframe::

       import pandas as pd

       iris_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv', nrows=100)

   Log it to Neptune::

        from neptunecontrib.api import log_table

        log_table('pandas_df', iris_df)

   Check out how the logged table looks in Neptune:
   https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-977/artifacts?path=tables%2F&file=pandas_df.html


