FAQ
===
.. _core-concepts_limits-top:

What is the API calls rate limit?
---------------------------------
`neptune-client <https://neptune.ai>`_ uses Python API to communicate with Neptune servers.
Users are restricted to 1 thousand requests per minute. If more requests are placed, neptune-client will retry sending the
data in the future (when usage does not approach the limit). In such case, users may notice some delay between the actual state of the
process that executes an experiment and data displayed in the Neptune Web UI. The extent of this effect is proportional
to the number of API calls over the 1 thousand limit.

.. note::

    Our experiences suggests that few AI research groups ever reach those limits.

.. External links

.. |Pricing| raw:: html

    <a href="https://neptune.ai/pricing" target="_blank">Pricing</a>

----

What happens if I lose my Internet connection during an experiment?
-------------------------------------------------------------------
Whenever you call a method from `neptune-client <https://neptune.ai>`_, your call is
translated into one or more API calls made over the network (the exception is when you're using `OfflineBackend`,
in which case, no network connectivity is required). Each such call may fail due to a number of reasons
(including lack of network connectivity). neptune-client handles such situations by retrying the request with
decreased frequency for the next 34 minutes.


Note that some of your method calls are translated to asychronous requests, which means that neptune-client processes
them in the background, while returning control to your code. Be aware that if the connectivity fails for a longer period -
say, 30 minutes, during which you push multiple requests to Neptune (for example, channel values), this might increase
your process's memory consumption. neptune-client applies no limits to this consumption.


So, if you lose Internet access for less than 34 minutes at a time and you don't send gigabytes of data during this time,
you will not lose any data - everything will be safely stored in Neptune.

What are the storage limits?
----------------------------
Please consult the |Pricing| page, where storage limits are listed.

----

What is the number of experiments or Notebooks limit?
-----------------------------------------------------
There are no limits. You can have as many experiments and Notebooks as you wish.
