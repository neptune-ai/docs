.. _integrations-sklearn:

Neptune-Sklearn Integration
===========================
|colab-script-neptune|

What will you get with this integration?
----------------------------------------
|sklearn-tour-loom|

|sklearn| is an open source machine learning framework commonly used for building predictive models. Neptune helps with keeping track of model training metadata.

With Neptune + Sklearn integration you can track your **classifiers**, **regressors** and **k-means** clustering results, specifically:

* log classifier and regressor parameters,
* log pickled model,
* log test predictions,
* log test predictions probabilities,
* log test scores,
* log classifier and regressor visualizations, like confusion matrix, precision-recall chart and feature importance chart,
* log KMeans cluster labels and clustering visualizations,
* log metadata including git summary info.

.. tip::
    You can log many other experiment metadata like interactive charts, video, audio and more.
    See the :ref:`full list <what-you-can-log>`.

.. note::

    This integration is tested with ``scikit-learn==0.23.2``, ``neptune-client==0.4.130``.

Where to start?
---------------
To get started with this integration:

* for classifier and regressor cases, follow the :ref:`Quickstart: classifier and regressor <sklearn-quickstart-reg-cls>`,
* for k-means clustering case, go here: :ref:`Quickstart: k-means <sklearn-quickstart-k-means>`.

You can also go to the demonstration of the selected convenience functions available in the :ref:`more options <sklearn-more-options>` section.

If you want to try things out and focus only on the code you can either:

|colab-script-neptune|

Before you start
----------------
You have ``Python 3.x`` and following libraries installed:

* ``neptune-client``. See :ref:`neptune-client installation guide <installation-neptune-client>`.

* ``scikit-learn``. See |scikit-install|.

.. code-block:: bash

    pip install scikit-learn

You also need minimal familiarity with scikit-learn. Have a look at this |scikit-guide| to get started.

.. _sklearn-quickstart-reg-cls:

Quickstart: classifier and regressor
------------------------------------
This quickstart will show you how to:

* Install the necessary Neptune and scikit-learn packages,
* Create the first experiment in project,
* Log trained regressor or classifier summary info to Neptune,
* Explore results in the Neptune UI.

Step 0: Create and fit example regressor or classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Prepare fitted regressor or classifier that will be further used in this quickstart. Below snippets show the idea:

**Classifier**

.. code-block:: python3

    gbc = GradientBoostingClassifier()

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    gbc.fit(X_train, y_train)

**Regressor**

.. code-block:: python3

    rfr = RandomForestRegressor()

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rfr.fit(X_train, y_train)

Both ``gbc`` (classification) and ``rfr`` (regression) objects will be later used to log various metadata to the experiment.

.. note::

    For this quickstart pick just one: classifier or regressor. In this way you will log only classifier/regressor results to the experiment. We do not want to mix results from these two :)

Step 1: Initialize Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^
Add the following snippet at the top of your script.

.. code-block:: python3

    import neptune

    neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/sklearn-integration')

.. tip::

    You can also use your personal API token. Read more about how to :ref:`securely set the Neptune API token <how-to-setup-api-token>`.

Step 2: Create an experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run the code below to create a Neptune experiment:

.. code-block:: python3

    neptune.create_experiment('sklearn-quickstart')

This also creates a link to the experiment. Open the link in a new tab.
The experiment will currently be empty, but keep the window open. You will be able to see estimator summary there.

When you create an experiment Neptune will look for the ``.git`` directory in your project and get the last commit information saved.

.. note::

    If you are using ``.py`` scripts for training Neptune will also log your training script automatically.

Step 3: Log estimator summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Log classifier or regressor summary to Neptune, by using :meth:`~neptunecontrib.monitoring.sklearn.log_regressor_summary` or :meth:`~neptunecontrib.monitoring.sklearn.log_classifier_summary`.

**Classification**

.. code-block:: python3

    from neptunecontrib.monitoring.sklearn import log_classifier_summary

    log_classifier_summary(gbc, X_train, X_test, y_train, y_test)

**Regression**

.. code-block:: python3

    from neptunecontrib.monitoring.sklearn import log_regressor_summary

    log_regressor_summary(rfr, X_train, X_test, y_train, y_test)

Step 4: See results in Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once data is logged you can switch to the Neptune tab which you had opened previously to explore results. Depending on your choice (classifier/regressor), you can check:

**Classifier**

* |cls-parameters|,
* |cls-model|,
* |cls-test-preds|,
* |cls-test-preds-proba|,
* |cls-test-scores|,
* |cls-visualizations| - look for "charts_sklearn",
* |cls-metadata| including git summary info.

|example-charts-classification|

**Regressor**

* |reg-parameters| as properties,
* |reg-model|,
* |reg-test-preds|,
* |reg-test-scores|,
* |reg-visualizations| - look for "charts_sklearn",
* |reg-metadata| including git summary info.

|example-charts-regression|

You can go to the |reference-documentation| to learn more. Remember that you can try it out with zero setup:

|colab-script-neptune|

.. _sklearn-quickstart-k-means:

Quickstart: K-Means
-------------------






.. _sklearn-more-options:

More Options
------------










Remember that you can try it out with zero setup:

|colab-script-neptune|

How to ask for help?
--------------------
Please visit the :ref:`Getting help <getting-help>` page. Everything regarding support is there.

Other integrations you may like
-------------------------------
You may also like these two integrations:

- :ref:`Optuna <integrations-optuna>`
- :ref:`Plotly <integrations-plotly>`


.. External links

.. |sklearn| raw:: html

    <a href="https://scikit-learn.org/stable/" target="_blank">scikit-learn</a>

.. |scikit-install| raw:: html

    <a href="https://scikit-learn.org/stable/install.html" target="_blank">scikit-learn installation guide</a>

.. |scikit-guide| raw:: html

    <a href="https://scikit-learn.org/stable/user_guide.html" target="_blank">scikit-learn guide</a>

.. |cls-parameters| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-312/details" target="_blank">logged classifier parameters</a>

.. |cls-model| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-312/artifacts?path=model%2F&file=estimator.skl" target="_blank">logged pickled model</a>

.. |cls-test-preds| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-312/artifacts?path=csv%2F&file=test_predictions.csv" target="_blank">logged test predictions</a>

.. |cls-test-preds-proba| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-312/artifacts?path=csv%2F&file=test_preds_proba.csv" target="_blank">logged test predictions probabilities</a>

.. |cls-test-scores| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-312/charts" target="_blank">logged test scores</a>

.. |cls-visualizations| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-312/logs" target="_blank">logged classifier visualizations</a>

.. |cls-metadata| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-312/details" target="_blank">logged metadata</a>

.. |example-charts-classification| raw:: html

    <div class="see-in-neptune">
        <a target="_blank"  href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-312/artifacts?path=csv%2F">
            <img width="50" height="50"
                src="https://neptune.ai/wp-content/uploads/neptune-ai-blue-vertical.png">
            <span>See example in Neptune</span>
        </a>
    </div>

.. |reg-parameters| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-311/details" target="_blank">logged regressor parameters</a>

.. |reg-model| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-311/artifacts?path=model%2F&file=estimator.skl" target="_blank">logged pickled model</a>

.. |reg-test-preds| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-311/artifacts?path=csv%2F&file=test_predictions.csv" target="_blank">logged test predictions</a>

.. |reg-test-scores| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-311/charts" target="_blank">logged test scores</a>

.. |reg-visualizations| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-311/logs" target="_blank">logged regressor visualizations</a>

.. |reg-metadata| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-311/details" target="_blank">logged metadata</a>

.. |example-charts-regression| raw:: html

    <div class="see-in-neptune">
        <a target="_blank"  href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-311/logs">
            <img width="50" height="50"
                src="https://neptune.ai/wp-content/uploads/neptune-ai-blue-vertical.png">
            <span>See example in Neptune</span>
        </a>
    </div>

.. |colab-script-neptune| raw:: html

    <div class="run-on-colab">

        <a target="_blank" href="https://colab.research.google.com//github/neptune-ai/neptune-examples/blob/master/integrations/sklearn/docs/Neptune-Scikit-learn.ipynb">
            <img width="50" height="50" src="https://neptune.ai/wp-content/uploads/colab_logo_120.png">
            <span>Run in Google Colab</span>
        </a>

        <a target="_blank" href="https://github.com/neptune-ai/neptune-examples/blob/master/integrations/sklearn/docs/Neptune-Scikit-learn.py">
            <img width="50" height="50" src="https://neptune.ai/wp-content/uploads/GitHub-Mark-120px-plus.png">
            <span>View source on GitHub</span>
        </a>
        <a target="_blank" href="https://ui.neptune.ai/o/shared/org/sklearn-integration/e/SKLEARN-632/charts">
            <img width="50" height="50" src="https://neptune.ai/wp-content/uploads/neptune-ai-blue-vertical.png">
            <span>See example in Neptune</span>
        </a>
    </div>

.. |sklearn-tour-loom| raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/3b2b03255f174223b4f3c55549892401" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

.. |reference-documentation| raw:: html

    <a href="https://docs.neptune.ai/api-reference/neptunecontrib/monitoring/sklearn/index.html" target="_blank">reference documentation</a>
