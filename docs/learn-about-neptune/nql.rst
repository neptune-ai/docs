Neptune Query Language (NQL)
============================
.. _core-concepts_nql:

Introduction
------------
Neptune Query Language (NQL) is a tool that enables you to apply complex filters to your experiments view (example below).

.. figure:: ../_static/images/others/nql_01.png
   :target: ../_static/images/others/nql_01.png
   :alt: experiments view with advanced search

Tutorial
--------
Let's assume that you want to see experiments where ``precision`` metric is higher than ``0.9``.
In other words, you are looking for experiments, where:

.. code-block:: mysql

    precision > 0.9

The condition above is a basic example of NQL.

----

Statement above is called *clause* and follows the following format (see :ref:`NQL reference <core-concepts_nql_reference>` for details):

.. code-block:: mysql

    field-name operator value

Note that it is required for field name to bo on the left side of an operator.

Now, imagine that a single clause is not enough, since you are looking for experiments where ``precision`` metric is higher than ``0.9``
and, at the same time, ``learning_rate`` parameter is smaller or equal ``0.005``:

.. code-block:: mysql

    precision > 0.9 AND learning_rate <= 0.005

In this example two clauses are joined together using logical operator (check :ref:`NQL reference <core-concepts_nql_reference>` for more details).

In a similar way you can build more complex queries.
The example below yields experiments where ``precision`` metric is higher than ``0.9`` and at least one of two conditions is satisfied:
either ``learning_rate`` parameter is smaller or equal ``0.005``, or ``encoder`` (a text log) is ``ResNet101``.

.. code-block:: mysql

    precision > 0.9 AND (learning_rate <= 0.005 OR encoder = ResNet101)

Advanced examples
-----------------
Fetching specific experiments by ids:

.. code-block:: mysql

    id = SAN-3 OR id = SAN-5 OR id = SAN-43

Complex logical experession:

.. code-block:: mysql

    ((param1 = 5 AND precision >= 0.9) OR (param1 < 5 AND param1 > 2 AND precision >= 0.7)) AND owner = Fred AND NOT status = Succeeded

Fetching experiments containing three specific tags:

.. code-block:: mysql

    tags CONTAINS some_tag_1 AND tags CONTAINS some_tag_2 AND tags CONTAINS another_tag

Fetching experiments containing at least one of specific tags:

.. code-block:: mysql

    tags CONTAINS some_tag_1 OR tags CONTAINS some_tag_2 OR tags CONTAINS another_tag

Fetching experiments containing tag ``expected`` but not containing tag ``unexpected``:

.. code-block:: mysql

    tags CONTAINS expected AND NOT tags CONTAINS unexpected

NQL reference
-------------
.. _core-concepts_nql_reference:

Clause
^^^^^^
Clause consists of three elements:

.. code-block:: mysql

    field-name operator value

**Field-name**

Field-names are case insensitive, so you can write both *state* and *State* or even *STATE*.
It can be one of the following:

* ``metric`` name

  Only last value in the metric is taken into account.

  Example:

  .. code-block:: mysql

      precision > 0.9

* ``parameter`` name

  Example:

  .. code-block:: mysql

      learning_rate <= 0.005

* ``tags``

  Can be used only with the ``CONTAINS`` operator. Condition is fulfilled if experiment contains a specific tag.

  Example:

  .. code-block:: mysql

      tags CONTAINS example-tag

* ``property`` name

  Example:

  .. code-block:: mysql

      train_data_path = "data/train.csv"
      train_data_path = train.csv

* ``text log`` name

  Only last value in the log is taken into account.

  Example:

  .. code-block:: mysql

      stderr = "example text in log file"

* ``id``

  Example:

  .. code-block:: mysql

      id = SAN-12

* ``state``

  The following values are possible for this field:

    - ``running``
    - ``succeeded``
    - ``aborted``
    - ``failed``

  Values of this field are case insensitive.

  Examples:

  .. code-block:: mysql

      state = running
      state = failed
      state = aborted

* ``owner``

  Example:

  .. code-block:: mysql

      owner = Fred

* ``name``

  Example:

  .. code-block:: mysql

      name = Approach-1

* ``description``

  Example:

  .. code-block:: mysql

      description = "My first experiment"

* ``size``

  Without any unit bytes are assumed, however following units are supported and are case insensitive: ``kb``, ``mb``, ``gb``.
  If there is a space between the number and its unit, the whole value needs to be enclosed in quotation marks.
  Comparison of this field works on its corresponding value, not on strings.

  Examples:

  .. code-block:: mysql

      size > 20MB
      size < 100
      size >= "35 kb"

* ``hostname``

  Example:

  .. code-block:: mysql

      hostname = my-server-1

----

**Operator**

It is one of the relational operators that let's you specify what you look for.
See the :ref:`operators table <core-concepts_nql_operators_reference>` below for list of all operators.

----

**Value**

Value is a specific value within given column, like ``0.95`` or ``ResNet101``. Values are case sensitive.
Two types of values are supported:

* numbers
* strings

Numbers are compared based on its values, however strings are compared lexicographically basing on ASCII codes.
Some fields, like ``size`` and ``state`` are exceptions to this rule.

Complex query
^^^^^^^^^^^^^^^
**AND and OR operators**

NQL query consists of a number of clauses connected with logical operators. For example:

.. code-block:: mysql

    precision > 0.9 AND learning_rate <= 0.005 AND encoder = ResNet101

Additionally brackets can be used to control logical operators precedence:

.. code-block:: mysql

    precision > 0.9 AND (learning_rate <= 0.005 OR encoder = ResNet101)

Notice: ``AND`` operator has a higher precedence than ``OR`` so two following queries are identical:


.. code-block:: mysql

    learning_rate <= 0.005 OR encoder = ResNet101 AND precision > 0.9
    learning_rate <= 0.005 OR (encoder = ResNet101 AND precision > 0.9)

**NOT operator**

There is also a ``NOT`` operator which can be used to negate a single clause or a whole sub-query.
For example if you want to find all experiments which are not owned by Fred you can use either of the following queries:

.. code-block:: mysql

    NOT owner = Fred
    owner != Fred

``NOT`` operator has higher precedence then ``AND`` and ``OR``, but lower precedence then relational operators.
So following queries are equal:

.. code-block:: mysql

    precision > 0.9 AND NOT learning_rate <= 0.005 OR encoder = ResNet101
    precision > 0.9 AND NOT (learning_rate <= 0.005) OR encoder = ResNet101
    precision > 0.9 AND (NOT learning_rate <= 0.005) OR encoder = ResNet101

but they are different from:

.. code-block:: mysql

    precision > 0.9 AND NOT (learning_rate <= 0.005 OR encoder = ResNet101)

Logical operators are case insensitive.

Operators reference
^^^^^^^^^^^^^^^^^^^
.. _core-concepts_nql_operators_reference:

==================== ===============================================================
Syntax elements
==================== ===============================================================
Logical operators    ``AND``, ``OR``, ``NOT``
Relational operators ``=``, ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``, ``CONTAINS``
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

.. code-block:: mysql

    owner = Fred


Quotes
^^^^^^

There are two types of quotation marks in NQL: ``""`` and ``````:

* A double quote (``""``) is used with values,
* back quote (``````) is used with field-names.

While in most cases it is not required to use quotation marks, there are some cases when it is necessary. See below.

**Special characters**

Typically, field name and string values can consist of letters of English alphabet, digits, dots (``.``), underscores (``_``) and dashes (``-``).
However, it is possible to write a query using strings containing any unicode character. For this purpose you will need to use quotation marks:

.. code-block:: mysql

    name = "my first experiment"

    `!@#$%^&*()_+` <= 0.005

    tags CONTAINS "Déjà vu"


.. note::

    If your field name contains a back quote character (`````) you will need to escape it using a backslash (``\``).
    Similarly, double quote character (``"``) has to be escaped in case of quote enclosed string value.
    Backslash character has to be preceded by another backslash in both cases - field names nad string values. For example:

    .. code-block:: mysql

        windows_path = "tmp\\dir\\file"
        text_with_quote = "And then he said: \"Hi!\""
        `\`backquoted_parameter_name\`` > 55
        `long\\parameter\\name\\with\\backslashes` > 55

**Keywords**

There are four reserved keywords in NQL: ``AND``, ``OR``, ``NOT`` and ``CONTAINS``.
They can not be simply used as fields or values.
Execution of one of the following queries will result in a syntax error:

.. code-block:: mysql

    AND = some_string
    name = CONTAINS
    tags CONTAINS CONTAINS

You can handle such situations by escaping the name of the column with back quotes (`````) and value of the field with quotes (``"``).

.. code-block:: mysql

    `AND` = some_string
    name = "CONTAINS"
    tags CONTAINS "CONTAINS"
