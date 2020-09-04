:mod:`neptune.model`
====================

.. py:module:: neptune.model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptune.model.ChannelWithLastValue
   neptune.model.LeaderboardEntry
   neptune.model.Point
   neptune.model.Points



.. py:class:: ChannelWithLastValue(channel_with_value_dto)

   Bases: :class:`object`

   .. attribute:: id
      

      

   .. attribute:: name
      

      

   .. attribute:: type
      

      

   .. attribute:: x
      

      

   .. attribute:: trimmed_y
      

      

   .. attribute:: y
      

      


.. py:class:: LeaderboardEntry(project_leaderboard_entry_dto)

   Bases: :class:`object`

   .. attribute:: id
      

      

   .. attribute:: name
      

      

   .. attribute:: state
      

      

   .. attribute:: internal_id
      

      

   .. attribute:: project_full_id
      

      

   .. attribute:: system_properties
      

      

   .. attribute:: channels
      

      

   .. attribute:: channels_dict_by_name
      

      

   .. attribute:: parameters
      

      

   .. attribute:: properties
      

      

   .. attribute:: tags
      

      

   .. method:: add_channel(self, channel)



.. py:class:: Point(point_dto)

   Bases: :class:`object`

   .. attribute:: x
      

      

   .. attribute:: numeric_y
      

      


.. py:class:: Points(point_dtos)

   Bases: :class:`object`

   .. attribute:: xs
      

      

   .. attribute:: numeric_ys
      

      


