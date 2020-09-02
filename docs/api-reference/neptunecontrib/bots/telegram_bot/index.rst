:mod:`neptunecontrib.bots.telegram_bot`
=======================================

.. py:module:: neptunecontrib.bots.telegram_bot

.. autoapi-nested-parse::

   Spins of a Neptune bot with which you can interact on telegram

   You can see which experiments are running, check the best experiements based
   on defined metric and even plot it in Telegram.

   Full list of options:
    * /project list NAMESPACE
    * /project select NAMESPACE/PROJECT_NAME
    * /project help
    * /experiments last NUMBER_OF_EXPERIMENTS
    * /experiments best METRIC_NAME NUMBER_OF_EXPERIMENTS
    * /experiments state STATE NUMBER_OF_EXPERIMENTS
    * /experiments help
    * /experiment link SHORT_ID
    * /experiment plot SHORT_ID METRIC_NAME OTHER_METRIC_NAME
    * /experiment help

   .. attribute:: telegram_api_token

      Your telegram bot api token.
      You can pass it either as --telegram_api_token or -t.

      :type: str

   .. attribute:: neptune_api_token

      Your neptune api token. If you
      set the NEPTUNE_API_TOKEN environemnt variable, you
      don't have to pass it here.
      You can pass it either as --neptune_api_token or -n.
      Default None.

      :type: str

   .. rubric:: Example

   Spin off your bot::

       $ python neptunecontrib.bots.telegram
           --telegram_api_token 'a1249auscvas0vbia0fias0'
           --neptune_api_token 'asdjpsvdsg987das0f9sad0fjasdf='

   Go to your telegram and type.

   `/project list neptune-ai`

   Use help to see what is implemented.

    * '/project help'
    * '/experiments help'
    * '/experiemnt help'



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptunecontrib.bots.telegram_bot.TelegramBot



Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.bots.telegram_bot.parse_args


.. py:class:: TelegramBot(telegram_api_token, neptune_api_token)

   .. method:: run(self)


   .. method:: project(self, bot, update, args)


   .. method:: experiments(self, bot, update, args)


   .. method:: experiment(self, bot, update, args)


   .. method:: unknown(self, bot, update)


   .. method:: _project_list(self, bot, update, args)


   .. method:: _project_select(self, bot, update, args)


   .. method:: _project_help(self, bot, update)


   .. method:: _experiments_last(self, bot, update, args)


   .. method:: _experiments_best(self, bot, update, args)


   .. method:: _experiments_state(self, bot, update, args)


   .. method:: _experiments_help(self, bot, update)


   .. method:: _experiment_link(self, bot, update, args)


   .. method:: _experiment_plot(self, bot, update, args)


   .. method:: _experiment_help(self, bot, update)


   .. method:: _no_project_selected(self, bot, update)



.. function:: parse_args()


.. data:: arguments
   

   

