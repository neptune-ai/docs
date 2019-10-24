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

Notice that it is required for field name to bo on the left side of an operator,
so it is not possible to compare values of two different columns.

Now, imagine that it's not enough, since you look for experiments where ``Precision`` metric is higher than ``0.9``
and, at the same time, ``learning_rate`` parameter is smaller or equal ``0.005``:

.. code-block:: sql

    Precision > 0.9 AND learning_rate <= 0.005

In this example two clauses are joined together using logical operator (check :ref:`NQL reference <core-concepts_nql_reference>` for more details).

In similar way you can build more complex queries.
Example below will return experiments where ``Precision`` metric is higher than ``0.9`` and one of two conditions is satisfied:
``learning_rate`` parameter is smaller or equal ``0.005``, or ``encoder`` (text log) is ``ResNet101``.

.. code-block:: sql

    Precision > 0.9 AND (learning_rate <= 0.005 OR encoder = ResNet101)


Advanced examples
-----------------

Fetching specific experiments by its ids:

.. code-block:: sql

    id = SAN-3 OR id = SAN-5 or id = SAN-43

Complex logical experession:

.. code-block:: sql

    ((param1 = 5 AND precision >= 0.9) OR (param1 < 5 AND param1 > 2 AND precision >= 0.7)) AND owner = Fred AND NOT result = error

Fetching experiments containing three specific tags:

.. code-block:: sql

    tags contains some_tag_1 AND tags contains some_tag_2 AND tags contains another_tag

Fetching experiments containing at least one of specific tags:

.. code-block:: sql

    tags contains some_tag_1 OR tags contains some_tag_2 OR tags contains another_tag

Fetching experiments containing tag ``expected`` but not containing tag ``unexpected``:

.. code-block:: sql

    tags contains expected AND NOT tags contains unexpected


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

1. ``field-name`` is one of the following:

  * ``id``
      Example:

      .. code-block:: sql

          id = SAN-12

  * ``state``
      Comparison of this field is not lexicographical.
      Instead the following order is defined on possible values of this field:

        - ``running``
        - ``succeeded``
        - ``aborted``
        - ``failed``

      Values of this field are case insensitive.

      Examples:

      .. code-block:: sql

          state = running
          state = Failed
          state = ABORTED

  * ``owner``
      Example:

      .. code-block:: sql

          owner = Fred

  * ``name``
      Example:

      .. code-block:: sql

          name = Untitled

  * ``description``
      Example:

      .. code-block:: sql

          description = "My first experiment"

  * ``size``
      Without any unit bytes are assumed, however following units are supported and are case insensitive: ``kb``, ``mb``, ``gb``.
      If there is a space between a number and a unit, a whole value have to be enclosed in quotation marks.
      Comparison of this field works on its corresponding value, not on strings.

      Examples:

      .. code-block:: sql

          size > 20MB
          size < 100
          size >= "35 kb"

  * ``hostname``
      Example:

      .. code-block:: sql

          hostname = salve-01

  * ``tags``
      Can be used only with ``contains`` operator. Condition is fulfilled if experiment contains a specific tag.

      Example:

      .. code-block:: sql

          tags contains test
  * parameter name
      Example:

      .. code-block:: sql

          learning_rate <= 0.005
  * metric name
      Only last value in the metric is taken into account.

      Example:

      .. code-block:: sql

          precision > 0.9
  * text log name
      Only last value in the log is taken into account.

      Example:

      .. code-block:: sql

          stderr = "ERROR: Currupted input data"
  * property name
      Example:

      .. code-block:: sql

          train_data_path = "data/train.csv"
          train_data_path = train.csv

  Notice that field names are case insensitive, so you can write both *state* and *State* or even *STATE*.
  However, with some exceptions (like `state`), values are case sensitive.

2. ``operator`` is on of relational operators that let's you specify what you look for.
   Check table below for list of all operators.

3. ``value`` is a specific value within given column, like ``0.95`` or ``ResNet101``.
   Two types of values are supported: numbers and strings.
   A type is is guessed basing on a field name and matters is case of comparison operators: ``>``, ``>=``, ``<``, ``<=``.
   Numbers are compared basing on its values, however strings are compared lexicographically basing on ASCII codes.
   Some fields, like ``size`` and ``state`` are exceptions to this rule.


Complex queries
^^^^^^^^^^^^^^^

**AND and OR operators**

NQL query consists of a number of clauses connected with logical operators. For exmpale:

.. code-block:: sql

    Precision > 0.9 AND learning_rate <= 0.005 AND encoder = ResNet101

Additionally brackets can be used to control logical operators precedence:

.. code-block:: sql

    Precision > 0.9 AND (learning_rate <= 0.005 OR encoder = ResNet101)

Notice: ``AND`` operator has a higher precedence than ``OR`` so two following queries are identical:


.. code-block:: sql

    learning_rate <= 0.005 OR encoder = ResNet101 AND Precision > 0.9
    learning_rate <= 0.005 OR (encoder = ResNet101 AND Precision > 0.9)

**NOT operator**

There is also a ``NOT`` operator which can be use to negate a single clause or a whole sub-query.
For example if you want to find all experiments which are not owned by Fred you can use one of the following queries:

.. code-block:: sql

    NOT owner = Fred
    owner != Fred

``NOT`` operator has higher precedence then ``AND`` and ``OR``, but lower precedence then relational operators.
So following queries are equal:

.. code-block:: sql

    Precision > 0.9 AND NOT learning_rate <= 0.005 OR encoder = ResNet101
    Precision > 0.9 AND NOT (learning_rate <= 0.005) OR encoder = ResNet101
    Precision > 0.9 AND (NOT learning_rate <= 0.005) OR encoder = ResNet101

but they both are different then:

.. code-block:: sql

    Precision > 0.9 AND NOT (learning_rate <= 0.005 OR encoder = ResNet101)

Logical operators are case insensitive.


Syntax reference
^^^^^^^^^^^^^^^^

==================== ===============================================================
Syntax elements
==================== ===============================================================
Logical operators    ``and``, ``or``, ``not``
Relational operators ``=``, ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``, ``contains``
Brackets             ``(``, ``)``
Quotation marks      ``""``, ``````
==================== ===============================================================

Precedence order
^^^^^^^^^^^^^^^^

If there are any field name collisions the following order precedence is applied:

  * system column
  * parameter
  * metric
  * text log
  * property

For example, if there is a metric and parameter called ``owner``, a following query will return only experiments
created by Fred, but no experiments of other users which have parameter called ``owner`` with value ``Fred``:

.. code-block:: sql

    owner = Fred


Quotes
^^^^^^

There are two types of quotation marks in NQL: ``""`` and ``````.
A double quote (``""``) is used with values and a back quote (``````) is used with field identifiers.
While in most cases it is not required to use quotation marks, there are some cases when it is necessary.

**Special characters**

Typically, field name and string values can consist of letters of english alphabet, digits, dots (``.``), underscores (``_``) and dashes (``-``).
However it is possible to write a query using strings containing any unicode character. For this purpose you will need to use quotation marks:

.. code-block:: sql

    name = "my first experiment"
    `!@#$%^&*()_+` <= 0.005
    tags contains "Déjà vu"

Notice: if your field name contains a back quote character (`````) you will need to escape it using a backslash (``\``).
Similarly, double quote character (``"``) have to be escaped in case of quote enclosed string value.
Backslash character have to be preceded by another backslash in both cases - field names nad string values. For example:

.. code-block:: sql

    windows_path = "tmp\\dir\\file"
    text_with_quote = "And then he said: \"Hi!\""
    `\`backquoted_parameter_name\`` > 55
    `long\\parameter\\name\\with\\backslashes` > 55

**Keywords**

There are four keywords in NQL which has assigned some special meaning.
Therefore they can not be simply used as fields or values. These keywords are: ``and``, ``or``, ``not`` and ``contains``.
Try of execution one of the following queries will result in syntax error:

.. code-block:: sql

    AND = some_string
    verb = contains
    tags contains contains

If your experiment has a column with such name or it's possible value of some column you will have to use quotes. For example:

.. code-block:: sql

    `AND` = some_string
    verb = "contains"
    tags contains "contains"
