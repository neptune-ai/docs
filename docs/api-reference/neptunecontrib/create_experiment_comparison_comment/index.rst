:mod:`neptunecontrib.create_experiment_comparison_comment`
==========================================================

.. py:module:: neptunecontrib.create_experiment_comparison_comment

.. autoapi-nested-parse::

   Create a markdown file with an experiment comparison table.

   Get a diff between experimenst across metrics, parameters, and properties and save it to a file as a markdown table.

   .. attribute:: experiment_ids

      Experiment ids of experiments you would like to compare. It works only for 2 experiments.
      You can pass it either as --experiment_ids or -e. For example, --experiment_ids GIT-83 GIT-82.

      :type: list(str)

   .. attribute:: tag_names

      tags of experiments you would like to compare.
      It works of tags passed are unique to the experiments they belong to.
      You can pass it either as --tag_ids or -i. For example, --tag_ids a892ee0ds 09asajd902.

      :type: list(str)

   .. attribute:: api_token

      Neptune api token. If you have NEPTUNE_API_TOKEN environment
      variable set to your API token you can skip this parameter. You can pass it either as --neptune_api_token or -t.

      :type: str

   .. attribute:: project_name

      Full name of the project. E.g. "neptune-ai/neptune-examples",
      If you have PROJECT_NAME environment variable set to your Neptune project you can skip this parameter.
      You can pass it either as --project_name or -p.

      :type: str

   .. attribute:: filepath

      filepath of the output markdown file. You can pass it either as --filepath or -f.

      :type: str

   .. rubric:: Example

   Create a file, comparison.md, with a comparison table of experiments GIT-83 and GIT-82::

       $ python -m neptunecontrib.create_experiment_comparison_comment             --tag_ids a892ee0ds 09asajd902             --api_token ANONYMOUS             --project_name shared/neptune-actions             --filepath comment_body.md

   .. note::

      If you keep your neptune api token in the NEPTUNE_API_TOKEN environment variable
      you can skip the --api_token.
      If you keep your full neptune project name in the PROJECT_NAME environment variable
      you can skip the --project_name.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.create_experiment_comparison_comment.get_project
   neptunecontrib.create_experiment_comparison_comment.get_experiment_data_by_id
   neptunecontrib.create_experiment_comparison_comment.get_experiment_data_by_tag
   neptunecontrib.create_experiment_comparison_comment.find_experiment_diff
   neptunecontrib.create_experiment_comparison_comment.create_comment_markdown
   neptunecontrib.create_experiment_comparison_comment.main
   neptunecontrib.create_experiment_comparison_comment.parse_args


.. function:: get_project(arguments)


.. function:: get_experiment_data_by_id(arguments)


.. function:: get_experiment_data_by_tag(arguments)


.. function:: find_experiment_diff(df)


.. function:: create_comment_markdown(df, project_name)


.. function:: main(arguments)


.. function:: parse_args()


.. data:: args
   

   

