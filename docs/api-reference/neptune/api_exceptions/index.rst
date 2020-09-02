:mod:`neptune.api_exceptions`
=============================

.. py:module:: neptune.api_exceptions


Module Contents
---------------

.. py:exception:: NeptuneApiException

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: SSLError

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: ConnectionLost

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ServerError

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: Unauthorized

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: Forbidden

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: InvalidApiKey

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: NamespaceNotFound(namespace_name)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ProjectNotFound(project_identifier)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: PathInProjectNotFound(path, project_identifier)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: NotebookNotFound(notebook_id, project=None)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ExperimentNotFound(experiment_short_id, project_qualified_name)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ChannelNotFound(channel_id)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ExperimentAlreadyFinished(experiment_short_id)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ExperimentLimitReached

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: StorageLimitReached

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ExperimentValidationError

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ChannelAlreadyExists(experiment_short_id, channel_name)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ChannelDoesNotExist(experiment_short_id, channel_name)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.


.. py:exception:: ChannelsValuesSendBatchError(experiment_short_id, batch_errors)

   Bases: :class:`neptune.api_exceptions.NeptuneApiException`

   Common base class for all non-exit exceptions.

   .. staticmethod:: _format_error(error)



