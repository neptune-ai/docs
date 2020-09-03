:mod:`neptune.oauth`
====================

.. py:module:: neptune.oauth


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptune.oauth.NeptuneAuth
   neptune.oauth.NeptuneAuthenticator



Functions
~~~~~~~~~

.. autoapisummary::

   neptune.oauth._no_token_updater


.. py:class:: NeptuneAuth(session)

   Bases: :class:`requests.auth.AuthBase`

   Base class that all auth implementations derive from

   .. method:: __call__(self, r)


   .. method:: _add_token(self, r)


   .. method:: refresh_token_if_needed(self)


   .. method:: _refresh_token(self)



.. py:class:: NeptuneAuthenticator(auth_tokens, ssl_verify, proxies)

   Bases: :class:`bravado.requests_client.Authenticator`

   Authenticates requests.

   :param host: Host to authenticate for.

   .. method:: matches(self, url)

      Returns true if this authenticator applies to the given url.

      :param url: URL to check.
      :return: True if matches host and port, False otherwise.


   .. method:: apply(self, request)

      Apply authentication to a request.

      :param request: Request to add authentication information to.



.. function:: _no_token_updater()


