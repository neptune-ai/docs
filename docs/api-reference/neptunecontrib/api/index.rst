:mod:`neptunecontrib.api`
=========================

.. py:module:: neptunecontrib.api


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   chart/index.rst
   explainers/index.rst
   html/index.rst
   table/index.rst
   utils/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.api.log_chart
   neptunecontrib.api.log_explainer
   neptunecontrib.api.log_local_explanations
   neptunecontrib.api.log_global_explanations
   neptunecontrib.api.log_html
   neptunecontrib.api.log_table
   neptunecontrib.api.concat_experiments_on_channel
   neptunecontrib.api.extract_project_progress_info
   neptunecontrib.api.get_channel_columns
   neptunecontrib.api.get_parameter_columns
   neptunecontrib.api.get_property_columns
   neptunecontrib.api.get_system_columns
   neptunecontrib.api.strip_prefices
   neptunecontrib.api.log_pickle
   neptunecontrib.api.get_pickle
   neptunecontrib.api.get_filepaths
   neptunecontrib.api.pickle_and_log_artifact
   neptunecontrib.api.get_pickled_artifact


.. function:: log_chart(name, chart, experiment=None)

   Logs charts from matplotlib, plotly, bokeh, and altair to neptune.

   Plotly, Bokeh, and Altair charts are converted to interactive HTML objects and then uploaded to Neptune
   as an artifact with path charts/{name}.html.

   Matplotlib figures are converted optionally. If plotly is installed, matplotlib figures are converted
   to plotly figures and then converted to interactive HTML and uploaded to Neptune as an artifact with
   path charts/{name}.html. If plotly is not installed, matplotlib figures are converted to PNG images
   and uploaded to Neptune as an artifact with path charts/{name}.png

   :param name: | Name of the chart (without extension) that will be used as a part of artifact's destination.
   :type name: :obj:`str`
   :param chart:
                 | Figure from `matplotlib` or `plotly`. If you want to use global figure from `matplotlib`, you
                   can also pass reference to `matplotlib.pyplot` module.
   :type chart: :obj:`matplotlib` or :obj:`plotly` Figure
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
       neptune.create_experiment(name='experiment_with_charts')

   Create matplotlib figure and log it to Neptune::

       import matplotlib.pyplot as plt

       fig = plt.figure()
       x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
       plt.hist(x, bins=5)
       plt.show()

       from neptunecontrib.api import log_chart

       log_chart('matplotlib_figure', fig)

   Create Plotly chart and log it to Neptune::

       import plotly.express as px

       df = px.data.tips()
       fig = px.histogram(df, x="total_bill", y="tip", color="sex", marginal="rug",
                          hover_data=df.columns)
       fig.show()

       from neptunecontrib.api import log_chart

       log_chart('plotly_figure', fig)

   Create Altair chart and log it to Neptune::

       import altair as alt
       from vega_datasets import data

       source = data.cars()

       chart = alt.Chart(source).mark_circle(size=60).encode(
                       x='Horsepower',
                       y='Miles_per_Gallon',
                       color='Origin',
                       tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
       ).interactive()

       from neptunecontrib.api import log_chart

       log_chart('altair_chart', chart)

   Create Bokeh figure and log it to Neptune::

       from bokeh.plotting import figure

       p = figure(plot_width=400, plot_height=400)

       # add a circle renderer with a size, color, and alpha
       p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)

       from neptunecontrib.api import log_chart

       log_chart('bokeh_figure', p)

   .. note::

      Check out how the logged charts look in Neptune:
      `example experiment
      <https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-973/artifacts?path=charts%2F&file=bokeh_figure.html>`_


.. function:: log_explainer(filename, explainer, experiment=None)

   Logs dalex explainer to Neptune.

   Dalex explainer is pickled and logged to Neptune.

   :param filename: filename that will be used as an artifact's destination.
   :type filename: :obj:`str`
   :param explainer: an instance of dalex explainer
   :type explainer: :obj:`dalex.Explainer`
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
                    project_qualified_name='shared/dalex-integration')
       neptune.create_experiment(name='logging explanations')

   Train your model and create dalex explainer::

       ...
       clf.fit(X, y)

       expl = dx.Explainer(clf, X, y, label="Titanic MLP Pipeline")

       log_explainer('explainer.pkl', expl)

   .. note::

      Check out how the logged explainer looks in Neptune:
      `example experiment <https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-48/artifacts>`_


.. function:: log_local_explanations(explainer, observation, experiment=None)

   Logs local explanations from dalex to Neptune.

   Dalex explanations are converted to interactive HTML objects and then uploaded to Neptune
   as an artifact with path charts/{name}.html.

   The following explanations are logged: break down, break down with interactions, shap, ceteris paribus,
   and ceteris paribus for categorical variables. Explanation charts are created and logged with default settings.
   To log charts with custom settings, create a custom chart and use `neptunecontrib.api.log_chart`.
   For more information about Dalex go to `Dalex Website <https://modeloriented.github.io/DALEX/>`_.

   :param explainer: an instance of dalex explainer
   :type explainer: :obj:`dalex.Explainer`
   :param observation (: obj): an observation that can be fed to the classifier for which the explainer was created
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
                    project_qualified_name='shared/dalex-integration')
       neptune.create_experiment(name='logging explanations')

   Train your model and create dalex explainer::

       ...
       clf.fit(X, y)

       expl = dx.Explainer(clf, X, y, label="Titanic MLP Pipeline")

       new_observation = pd.DataFrame({'gender': ['male'],
                                       'age': [25],
                                       'class': ['1st'],
                                       'embarked': ['Southampton'],
                                       'fare': [72],
                                       'sibsp': [0],
                                       'parch': 0},
                                      index=['John'])

       log_local_explanations(expl, new_observation)

   .. note::

      Check out how the logged explanations look in Neptune:
      `example experiment <https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-48/artifacts?path=charts%2F>`_


.. function:: log_global_explanations(explainer, categorical_features=None, numerical_features=None, experiment=None)

   Logs global explanations from dalex to Neptune.

   Dalex explanations are converted to interactive HTML objects and then uploaded to Neptune
   as an artifact with path charts/{name}.html.

   The following explanations are logged: variable importance. If categorical features are specified partial dependence
   and accumulated dependence are also logged. Explanation charts are created and logged with default settings.
   To log charts with custom settings, create a custom chart and use `neptunecontrib.api.log_chart`.
   For more information about Dalex go to `Dalex Website <https://modeloriented.github.io/DALEX/>`_.

   :param explainer: an instance of dalex explainer
   :type explainer: :obj:`dalex.Explainer`
   :param categorical_features (: list): list of categorical features for which you want to create
                                  accumulated dependence plots.
   :param numerical_features (: list): list of numerical features for which you want to create
                                partial dependence plots.
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
                    project_qualified_name='shared/dalex-integration')
       neptune.create_experiment(name='logging explanations')

   Train your model and create dalex explainer::

       ...
       clf.fit(X, y)

       expl = dx.Explainer(clf, X, y, label="Titanic MLP Pipeline")
       log_global_explanations(expl, categorical_features=["gender", "class"], numerical_features=["age", "fare"])

   .. note::

      Check out how the logged explanations look in Neptune:
      `example experiment <https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-48/artifacts?path=charts%2F>`_


.. function:: log_html(name, html, experiment=None)

   Logs html to neptune.

   HTML is logged to Neptune as an artifact with path html/{name}.html

   :param name: | Name of the chart (without extension) that will be used as a part of artifact's destination.
   :type name: :obj:`str`
   :param html_body: | HTML string that is logged and rendered as HTML.
   :type html_body: :obj:`str`
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
       neptune.create_experiment(name='experiment_with_html')

   Create an HTML string::

       html = "<button type='button',style='background-color:#005879; width:300px; height:200px; font-size:30px'>                  <a style='color: #ccc', href='https://docs.neptune.ai'> Take me back to the docs!!<a> </button>"

   Log it to Neptune::

        from neptunecontrib.api import log_html

        log_html('go_to_docs_button', html)

   Check out how the logged table looks in Neptune:
   https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-988/artifacts?path=html%2F&file=button_example.html


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


.. function:: concat_experiments_on_channel(experiments, channel_name)

   Combines channel values from experiments into one dataframe.

   This function helps to compare channel values from a list of experiments
   by combining them in a dataframe. E.g: Say we want to extract the `log_loss`
   channel values for a list of experiments. The resulting dataframe will have
   ['id','x_log_loss','y_log_loss'] columns.

   :param experiments: list of `neptune.experiments.Experiment` objects.
   :type experiments: list
   :param channel_name: name of the channel for which we want to extract values.
   :type channel_name: str

   :returns: Dataframe of ['id','x_CHANNEL_NAME','y_CHANNEL_NAME']
             values concatenated from a list of experiments.
   :rtype: `pandas.DataFrame`

   .. rubric:: Examples

   Instantiate a session::

       from neptune.sessions import Session
       session = Session()

   Fetch a project and a list of experiments::

       project = session.get_projects('neptune-ai')['neptune-ai/Salt-Detection']
       experiments = project.get_experiments(state=['aborted'], owner=['neyo'], min_running_time=100000)

   Construct a channel value dataframe::

       from neptunecontrib.api.utils import concat_experiments_on_channel
       compare_df = concat_experiments_on_channel(experiments,'unet_0 epoch_val iout loss')

   .. note::

      If an experiment in the list of experiments does not contain the channel with a specified channel_name
      it will be omitted.


.. function:: extract_project_progress_info(leadearboard, metric_colname, time_colname='finished')

   Extracts the project progress information from the experiment view.

   This function takes the experiment view (leaderboard) and extracts the information
   that is important for analysing the project progress. It creates additional columns
   `metric` (actual experiment metric), `metric_best` (best metric score to date)),
   `running_time_day` (total amount of experiment running time for a given day in hours),
   'experiment_count_day' (total number of experiments ran in a given day).

   This function is usually used with the `plot_project_progress` from `neptunecontrib.viz.projects`.

   :param leadearboard: Dataframe containing the experiment view of the project.
                        It can be extracted via `project.get_leaderboard()`.
   :type leadearboard: `pandas.DataFrame`
   :param metric_colname: name of the column containing the metric of interest.
   :type metric_colname: str
   :param time_colname: name of the column containing the timestamp. It can be either `finished`
                        or `created`. Default is 'finished'.
   :type time_colname: str

   :returns: Dataframe of ['id', 'metric', 'metric_best', 'running_time',
             'running_time_day', 'experiment_count_day', 'owner', 'tags', 'timestamp', 'timestamp_day']
             columns.
   :rtype: `pandas.DataFrame`

   .. rubric:: Examples

   Instantiate a session::

       from neptune.sessions import Session
       session = Session()

   Fetch a project and the experiment view of that project::

       project = session.get_projects('neptune-ai')['neptune-ai/Salt-Detection']
       leaderboard = project.get_leaderboard()

   Create a progress info dataframe::

       from neptunecontrib.api.utils import extract_project_progress_info
       progress_df = extract_project_progress_info(leadearboard,
                                                   metric_colname='channel_IOUT',
                                                   time_colname='finished')


.. function:: get_channel_columns(columns)

   Filters leaderboard columns to get the channel column names.

   :param columns: Iterable of leaderboard column names.
   :type columns: iterable

   :returns: A list of channel column names.
   :rtype: list


.. function:: get_parameter_columns(columns)

   Filters leaderboard columns to get the parameter column names.

   :param columns: Iterable of leaderboard column names.
   :type columns: iterable

   :returns: A list of channel parameter names.
   :rtype: list


.. function:: get_property_columns(columns)

   Filters leaderboard columns to get the property column names.

   :param columns: Iterable of leaderboard column names.
   :type columns: iterable

   :returns: A list of channel property names.
   :rtype: list


.. function:: get_system_columns(columns)

   Filters leaderboard columns to get the system column names.

   :param columns: Iterable of leaderboard column names.
   :type columns: iterable

   :returns: A list of channel system names.
   :rtype: list


.. function:: strip_prefices(columns, prefices)

   Filters leaderboard columns to get the system column names.

   :param columns: Iterable of leaderboard column names.
   :type columns: iterable
   :param prefices: List of prefices to strip. You can choose one of
                    ['channel_', 'parameter_', 'property_']
   :type prefices: list

   :returns: A list of clean column names.
   :rtype: list


.. function:: log_pickle(filename, obj, experiment=None)

   Logs picklable object to Neptune.

   Pickles and logs your object to Neptune under specified filename.

   :param obj: Picklable object.
   :param filename: filename under which object will be saved to Neptune.
   :type filename: str
   :param experiment: Neptune experiment.
   :type experiment: `neptune.experiments.Experiment`

   .. rubric:: Examples

   Initialize Neptune::

       import neptune
       neptune.init('USER_NAME/PROJECT_NAME')

   Create RandomForest object and log to Neptune::

       from sklearn.ensemble import RandomForestClassifier
       from neptunecontrib.api import log_pickle

       neptune.create_experiment()

       rf = RandomForestClassifier()
       log_pickle('rf.pkl', rf)


.. function:: get_pickle(filename, experiment)

   Downloads pickled artifact object from Neptune and returns a Python object.

   Downloads the pickled object from artifacts of given experiment,
    loads it to memory and returns a Python object.

   :param filename: filename under which object will be saved to Neptune.
   :type filename: str
   :param experiment: Neptune experiment.
   :type experiment: `neptune.experiments.Experiment`

   .. rubric:: Examples

   Initialize Neptune::

       import neptune

       project = neptune.init('USER_NAME/PROJECT_NAME')

   Choose Neptune experiment::

       experiment = project.get_experiments(id=['PRO-101'])[0]

   Get your pickled object from experiment artifacts::

       from neptunecontrib.api import get_pickle

       results = get_pickle('results.pkl', experiment)


.. function:: get_filepaths(dirpath='.', extensions=None)

   Creates a list of all the files with selected extensions.

   :param dirpath: Folder from which all files with given extensions should be added to list.
   :type dirpath: str
   :param extensions: All extensions with which files should be added to the list.
   :type extensions: list(str) or None

   :returns: A list of filepaths with given extensions that are in the directory or subdirecotries.
   :rtype: list

   .. rubric:: Examples

   Initialize Neptune::

        import neptune
        from neptunecontrib.versioning.data import log_data_version
        neptune.init('USER_NAME/PROJECT_NAME')

   Create experiment and track all .py files from given directory and subdirs::

        with neptune.create_experiment(upload_source_files=get_filepaths(extensions=['.py'])):
            neptune.send_metric('score', 0.97)


.. function:: pickle_and_log_artifact(obj, filename, experiment=None)


.. function:: get_pickled_artifact(experiment, filename)


