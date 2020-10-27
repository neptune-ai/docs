.. _integrations-deepnote:

Deepnote
========

Step 1: Install neptune-client
!pip install neptune-client==0.4.123
Step 2: Set up an environment variable for the API token
Create a new environment variable integration in the left tab where you set the NEPTUNE_API_TOKEN. Alternatively, you can initialize neptune with the API token directly with the snippet:

# Alternative version to initialising Neptune
neptune.init(project_qualified_name='<name_here>',
              api_token='<token_here>',
             )
The package neptune-client is auto-installed from requirements.txt.

More information about the quick start with Neptune

Step 3: Replace the project name and log metrics into a Neptune dashboard
import neptune

# The init() function called this way assumes that
# NEPTUNE_API_TOKEN environment variable is defined by the integration.

neptune.init('<NEPTUNE_PROJECT_NAME>')
neptune.create_experiment(name='minimal_example')

# log some metrics

for i in range(100):
    neptune.log_metric('loss', 0.95**i)

neptune.log_metric('AUC', 0.96)