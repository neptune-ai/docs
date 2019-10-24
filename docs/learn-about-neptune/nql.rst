Neptune Query Language (NQL)
============================
.. _core-concepts_nql:

Introduction
------------
Neptune Query Language (NQL) is a tool for advanced search for experiments in the experiments view (example below).
The "type query" field above the table allows you to write complex queries (check syntax below) to search for experiments of interest.

.. figure:: ../_static/images/others/nql_01.png
   :target: ../_static/images/others/nql_01.png
   :alt: experiments view with advanced search

Basics with examples
--------------------
Let's assume that you want to see experiments where ``Precision`` metric is higher than ``0.9``.
In other words, you look for experiments, where:

.. code-block:: sql

    Precision > 0.9

This is basic example of the NQL.

----

Statement above is called *clause* and follows template (check :ref:`NQL reference <core-concepts_nql_reference>` for more details):

.. code-block:: sql

    field-name operator value

Now, imagine that it's not enough, since you look for experiments where ``Precision`` metric is higher than ``0.9``
and, at the same time, ``learning_rate`` parameter is smaller or equal ``0.005``:

.. code-block:: sql

    Precision > 0.9 AND learning_rate =< 0.005

In this example two clauses are joined together using logical operator (check :ref:`NQL reference <core-concepts_nql_reference>` for more details).

In similar way you can build more complex queries.
Example below will return experiments where ``Precision`` metric is higher than ``0.9`` and one of two conditions is satisfied:
``learning_rate`` parameter is smaller or equal ``0.005``, or ``encoder`` (text log) is ``ResNet101``.

.. code-block:: sql

    Precision > 0.9 AND (learning_rate =< 0.005 OR encoder = ResNet101)

Advanced examples
-----------------

ToDo:
- show tag, tags, numeric column, ranges, exp owner, system columns, parameters, properties
- explain how to use tags in query
- focus on working examples - give short explanation.

How does it relates to SQL queries?
-----------------------------------
Let's take first example into consideration, where you look for experiments where ``Precision`` metric is higher than ``0.9``.
In the SQL world, you may write:

.. code-block:: sql

    SELECT *
    FROM experiments_table
    WHERE Precision > 0.9;

Now you just drop this part of the SQL query: ``SELECT * FROM experiments_table WHERE``. You only care about the condition itself.


NQL reference
-------------
.. _core-concepts_nql_reference:

Clause
^^^^^^
Clause consist of three elements:

.. code-block:: sql

    field-name operator value

1. field-name is everything that you can see as a column in experiments view.
2. operator is a arithmetic operation that let's you specifiy what you look for. Check table below for list of all operators.
3. value is a specific value within given column, like ``0.95`` or ``ResNet101``.

Syntax reference
^^^^^^^^^^^^^^^^

================ ===========================
Syntax elements
================ ===========================
Logical          ``and`` ``or``
Arithmetical     ``=, ==, !=, >, =>, <, =<``
Brackets         ``(, )``
Special keywords ``contains``
================ ===========================

Precedence order
^^^^^^^^^^^^^^^^

ToDo
- elements of syntax
- columns in the leaderboard (if any)
