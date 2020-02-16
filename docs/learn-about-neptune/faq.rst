FAQ
===
.. _core-concepts_limits-top:

Q: What is the API calls rate limits?
-------------------------------------
`Neptune-client <https://neptune.ai>`_ uses Python API to communicate with Neptune servers. Users are restricted to 1k requests per minute. If more requests are being placed, neptune-client will retry sending the data in the future (when usage does not approach the limit). In such case, Users may notice some delay between the actual state of the process that executes an experiment and data displayed in Neptune Web application. Extent of this effect is proportional to the number of API calls over the 1k limit.

.. note::

    Our experiences suggests that only few AI research groups hit those limits.

.. External links

.. |pricing| raw:: html

    <a href="https://neptune.ai/pricing" target="_blank">pricing</a>

----

Q: What happens if I lose Internet connectivity during experiment?
------------------------------------------------------------------
Whenever you call a method from `Neptune-client <https://neptune.ai>`_, your call is translated into one or more API calls made over network (the exception is when you're using `OfflineBackend`, in which case no network connectivity is required). Each such call may fail due to a number of reasons (including lack of network connectivity). Neptune-client handles such situations by retrying the request with decreased frequency for the next 34 minutes.


Note that some of your method calls are translated to asychronous requests, which means that Neptune-client processes them in the background, while returning control to your code. Be aware, that if the connectivity fails for a longer period - say, 30 minutes, during which you push multiple requests to Neptune (e.g. channel values), this might increase your process' memory consumption. Neptune-client doesn't apply any limits on this consumption.


So, if you lose Internet access for less than 34 minutes at a time and you don't send gigabytes of data during this time, you will not lose any data - everything will be safely stored in Neptune.

Q: What are storage limits?
---------------------------
Please consult |pricing| page, where storage limits are listed.

----

Q: What is the number of experiments or Notebooks limit?
--------------------------------------------------------
There are no limits. You can have as many experiments and Notebooks as you wish.

