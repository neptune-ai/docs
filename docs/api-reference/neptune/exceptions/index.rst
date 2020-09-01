:mod:`neptune.exceptions`
=========================

.. py:module:: neptune.exceptions


Module Contents
---------------

.. py:exception:: NeptuneException

   Bases: :class:`Exception`

   Common base class for all non-exit exceptions.


.. py:exception:: Uninitialized

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: FileNotFound(path)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: NotAFile(path)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: NotADirectory(path)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: InvalidNotebookPath(path)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: InvalidChannelX(x)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: NoChannelValue

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: LibraryNotInstalled(library)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: InvalidChannelValue(expected_type, actual_type)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: NoExperimentContext

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: MissingApiToken

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: MissingProjectQualifiedName

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: IncorrectProjectQualifiedName(project_qualified_name)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: InvalidNeptuneBackend(provided_backend_name)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: DeprecatedApiToken(app_url)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: CannotResolveHostname(host)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


.. py:exception:: UnsupportedClientVersion(version, minVersion, maxVersion)

   Bases: :class:`neptune.exceptions.NeptuneException`

   Common base class for all non-exit exceptions.


