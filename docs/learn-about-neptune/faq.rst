FAQ
===
.. _core-concepts_limits-top:

Q: What are storage limits?
---------------------------
Please consult |pricing| page, where storage limits are listed.

----

Q: What is the number of experiments or notebooks limit?
--------------------------------------------------------
There are no limits. You can have as many experiments and notebooks as you wish.

----

Q: What is the API calls rate limits?
-------------------------------------
`Neptune-client <https://neptune.ai>`_ uses Python API to communicate with Neptune servers. Users are restricted to 1k requests per minute. If more requests are being placed, neptune-client will retry sending the data in the future (when usage does not approach the limit). In such case, Users may notice some delay between the actual state of the process that executes an experiment and data displayed in Neptune Web application. Extent of this effect is proportional to the number of API calls over the 1k limit.

.. note::

    Our experiences suggests that only few AI research groups hit those limits.

.. External links

.. |pricing| raw:: html

    <a href="https://neptune.ai/pricing" target="_blank">pricing</a>
