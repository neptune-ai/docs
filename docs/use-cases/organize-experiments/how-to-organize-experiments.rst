How to organize ML experimentation: step by step guide
======================================================

|run on colab button|

Introduction
------------

This guide will show you how to:

- Keep track of code, data, environment and parameters
- Log results like evaluation metrics and model files
- Find experiments in the experiment dashboard with tags
- Organize experiments in a dashboard view and save it for later

Before you start
----------------

Make sure you meet the following prerequisites before starting:

- Have Python 3.x installed
- Have scikit-learn and joblib installed
- |Have Neptune installed|
- |Create a project|
- |Configure Neptune API token on your system|

.. note::

    You can run this how-to on Google Colab with zero setup.

    Just click on the ``Open in Colab`` button on the top of the page.

Step 1: Create a basic training script
--------------------------------------

As an example I'll use a script that trains a sklearn model on wine dataset.

.. note::

    You **don't have to use sklearn** to track your training runs with Neptune.

    I am using it as an easy to follow example.

    There are links to integrations with other ML frameworks and useful articles in the text.

1. Create a file ``train.py`` and copy the script below.

``train.py``

.. code:: python

    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from joblib import dump

    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                        test_size=0.4, random_state=1234)

    params = {'n_estimators': 10,
              'max_depth': 3,
              'min_samples_leaf': 1,
              'min_samples_split': 2,
              'max_features': 3,
              }

    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict_proba(X_train)
    y_test_pred = clf.predict_proba(X_test)

    train_f1 = f1_score(y_train, y_train_pred.argmax(axis=1), average='macro')
    test_f1 = f1_score(y_test, y_test_pred.argmax(axis=1), average='macro')
    print(f'Train f1:{train_f1} | Test f1:{test_f1}')

    dump(clf, 'model.pkl')

2. Run training to make sure that it works correctly.

.. code:: bash

   python train.py

Step 3: Connect Neptune to your script
--------------------------------------

At the top of your script add

.. code:: python

    import neptune

    neptune.init(project_qualified_name='shared/onboarding',
                 api_token='ANONYMOUS',
                 )

You need to tell Neptune who you are and where you want to log things.

To do that you specify:

- ``project_qualified_name=USERNAME/PROJECT_NAME``: Neptune username and project
- ``api_token=YOUR_API_TOKEN``: your Neptune API token.

.. note::

    If you configured your Neptune API token correctly, as described in |Configure Neptune API token on your system|, you can skip ``api_token`` argument:

    .. code:: python

        neptune.init(project_qualified_name='YOUR_USERNAME/YOUR_PROJECT_NAME')

Step 4. Create an experiment and add parameter, code and environment tracking
-----------------------------------------------------------------------------

To start logging things to Neptune you need to create an experiment.
An experiment is an object to which you log various objects.

Some object types like parameters and source code can only be logged when you create experiment.

Let's go over that step-by-step.

1. Create an experiment

.. code:: python

    neptune.create_experiment(name='great-idea')

This opens a new "experiment" namespace in Neptune to which you can log various objects.
You can add ``name`` to your experiment but it's optional.

2. Add parameters tracking

.. code:: python

    neptune.create_experiment(params=params)

To log parameters you need to pass a dictionary to the ``params`` argument.

3. Add code and environment tracking

.. code:: python

    neptune.create_experiment(upload_source_files=['*.py', 'requirements.txt'])

You can log source code to Neptune with every experiment run.
It can save you if you forget to commit your code changes to git.

To do it pass a list of files or regular expressions to ``upload_source_files`` argument.

.. note::

    Neptune automatically finds the ``.git`` directoy and logs the git commit information like:

    - commit id sha
    - commit message
    - commit author email
    - commit datetime
    - whether the experiment is run on a dirty commit (code change but wasn't commited to git)

Putting it all together your ``neptune.create_experiment`` should look like this:

.. code:: python

    neptune.create_experiment(name='great-idea', # name experiment
                              params=params,  # log parameters
                              upload_source_files=['*.py', 'requirements.txt']  # log source and environment
                              )

Step 5. Add tags to organize things
-----------------------------------

.. code:: python

    neptune.append_tag(['experiment-organization', 'me'])  # organize things

Pass a list of strings to the ``.append_tag`` method of the experiment object.

It will help you find experiments later, especially if you try a lot of ideas.

.. note::

    You can also add tags at experiment creation via ``tags`` argument

    .. code:: python

        neptune.create_experiment(tags=['experiment-organization', 'me'])

Step 6. Add logging of train and evaluation metrics
---------------------------------------------------

.. code:: python

    neptune.log_metric('train_f1', train_f1)
    neptune.log_metric('test_f1', test_f1)

Log all the metrics you care about with ``.log_metric`` method. There could be as many as you like.
The first argument is the name of the metric, the second it's value.

.. note::

    You can log multiple values to the same metric. When you do that a chart will be created automatically.

Step 7. Add logging of model files
----------------------------------

.. code:: python

    neptune.log_artifact('model.pkl')

Log your model with ``.log_artifact`` method. Just pass the path to the file you want to log to Neptune.

.. note::

    You can also log picklable Python objects directly with |log_pickle function from neptune-contrib|.

    .. code:: python

        from neptunecontrib.api import log_pickle

        ...
        rf = RandomForestClassifier()
        log_pickle('rf.pkl', rf)

Step 8. Run a few experiments with different parameters
-------------------------------------------------------

Let's run some experiments with different model configuration.

1. Change parameters in the ``params`` dictionary

.. code:: python

    params = {'n_estimators': 10,
              'max_depth': 3,
              'min_samples_leaf': 1,
              'min_samples_split': 2,
              'max_features': 3,
              }

2. Run an experiment

.. code:: bash

    python train.py

Step 9. Go to Neptune UI
------------------------

Click on one of the links created when you run the script or go directly to the app.

|click on link|

If you created your own project in Neptune you can also go to projects tab and find it.

|user project|

If you are logging things to the public project ``shared/onboarding`` you can just |follow this link|.

Step 10. See that everything got logged
---------------------------------------

Go to one of the experiments you ran and see that you logged things correctly:

- click on the experiment link or one of the rows in the experiment table in the UI
- Go to ``Logs`` section to see your metrics
- Go to ``Source code`` to see that your code was logged
- Go to ``Artifacts`` to see that the model was saved

|See one experiment|

Step 11. Filter experiments by tag
----------------------------------

Go to the experiments space and:

1. Click on the ``go to simple search``
2. In the ``tags`` type ``experiment-organization`` to find it (or other tag you added to your experiment).
3. Select the tag.

Neptune should filter all those experiments for you.

|filter with tag|

Step 12. Choose parameter and metric columns you want to see
------------------------------------------------------------

Use the ``Manage columns`` button to choose the columns for the experiment table:

- Click on ``Manage columns``
- Go to the ``Numeric logs`` and ``Text parameters`` or type a name of your metric or parameter to find it.
- Add ``test_f1`` metric and the parameters you tweaked (in my case ``max_depth``, ``max_features``, ``min_samples_leaf``, and ``n_estimators``).

|manage columns|

.. tip::

    You can also use the suggested columns which shows you the columns with values that differ between selected experiments.

    Just click on the ``+`` to add it to your experiment table.

Step 13. Save the view of experiment table
------------------------------------------

You can save the current view of experiment table for later:

- Click on the ``Save as new``

Both the columns and the filtering on rows will be saved as view.

|save view|

.. tip::

    Create and save multiple views of the experiment table for different use cases or experiment groups.

What's next
-----------

Now that you know how to keep track of experiments and organize them you can:

- See |how to log other objects and monitor training in Neptune|
- See |how to connect Neptune to your codebase|
- |Check our integrations| with other frameworks

Full script
-----------

|run on colab button|

.. code:: python

    import neptune
    from joblib import dump
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    neptune.init(api_token='ANONYMOUS',
                 project_qualified_name='shared/onboarding')

    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                        test_size=0.4, random_state=1234)

    params = {'n_estimators': 10,
              'max_depth': 3,
              'min_samples_leaf': 1,
              'min_samples_split': 2,
              'max_features': 3,
              'random_state': 1234
              }

    neptune.create_experiment('great-idea',
                              params=params,  # log parameters
                              upload_source_files=['*.py', 'requirements.txt'],  # log source and environment
                              tags=['experiment-organization', 'me'])  # organize things

    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict_proba(X_train)
    y_test_pred = clf.predict_proba(X_test)

    train_f1 = f1_score(y_train, y_train_pred.argmax(axis=1), average='macro')
    test_f1 = f1_score(y_test, y_test_pred.argmax(axis=1), average='macro')
    print(f'Train f1:{train_f1} | Test f1:{test_f1}')

    neptune.log_metric('train_f1', train_f1)  # log metrics
    neptune.log_metric('test_f1', test_f1)  # log metrics

    dump(clf, 'model.pkl')
    neptune.log_artifact('model.pkl')  # log files


.. |Create a project| raw:: html

    <a href="/workspace-project-and-user-management/projects/create-project.html" target="_blank">Create a project in Neptune</a>

.. |Configure Neptune API token on your system| raw:: html

    <a href="/security-and-privacy/api-tokens/how-to-find-and-set-neptune-api-token.html" target="_blank">Configure Neptune API token on your system</a>

.. |Have Neptune installed| raw:: html

    <a href="/getting-started/installation/index.html">Have Neptune installed</a>

.. |run on colab button| raw:: html

    <a href="https://colab.research.google.com//github/neptune-ai/neptune-colab-examples/blob/master/Organize-ML-experiments.ipynb" target="_blank">
        <img width="200" height="200"src="https://colab.research.google.com/assets/colab-badge.svg"></img>
    </a>

.. |how to log other objects and monitor training in Neptune| raw:: html

    <a href="https://neptune.ai/blog/monitoring-machine-learning-experiments-guide" target="_blank">how to log other objects and monitor training in Neptune</a>

.. |how to connect Neptune to your codebase| raw:: html

    <a href="/getting-started/integrating-neptune-into-your-codebase/step-by-step-connect-neptune.html" target="_blank">how to connect Neptune to your codebase</a>

.. |Check our integrations| raw:: html

    <a href="/integrations/index.html" target="_blank">Check our integrations</a>

.. |log_pickle function from neptune-contrib|  raw:: html

    <a href="/api-reference/neptunecontrib/api/index.html?highlight=log_pickle#neptunecontrib.api.log_pickle" target="_blank">log_pickle function from neptune-contrib</a>

.. |follow this link| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/onboarding/e/ON-261" target="_blank">follow this link</a>

.. |click on link| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/6ztCBfYuDKA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |user project| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/rEC-sxhP72w" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |See one experiment| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/WpAq7Kj88ec" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |filter with tag| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/ppPOtU_lNkk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |manage columns| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/gvlIXa25-Bc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |save view| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/iTgjtYBWqko" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
