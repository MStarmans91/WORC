Development and Design Documentation
====================================

In this chapter we will discuss the design of Fastr in more detail. We give
pointers for development and add the design documents as we currently envision
Fastr. This is both for people who are interested in the Fastr develop and for
current developers to have an archive of the design decision agreed upon.

Sample flow in Fastr
--------------------

The current Sample flow is the following:

.. graphviz::

    digraph sampleflow {
       Output [
          shape=plaintext
          label=<
            <table border="0">
              <tr>
                <td border="1px" width="120" height="40" port="port">Output</td>
                <td border="0" width="140" height="40"><b>ContainsSamples</b></td>
                <td border="0" width="120" height="40" align="left"></td>
              </tr>
            </table>
          >
       ];
       SubOutput [
          shape=plaintext
          label=<
            <table border="0">
              <tr>
                <td border="1px" width="120" height="40" port="port">SubOutput</td>
                <td border="0" width="140" height="40"><b>ForwardsSamples</b></td>
                <td border="0" width="120" height="40" align="left">selects cardinality</td>
              </tr>
            </table>
          >
       ];
       Link [
          shape=plaintext
          label=<
            <table border="0">
              <tr>
                <td border="1px" width="120" height="40" port="port">Link</td>
                <td border="0" width="140" height="40"><b>ForwardsSamples</b></td>
                <td border="0" width="120" height="40" align="left">collapse + expand (changes cardinality and dimensions)</td>
              </tr>
            </table>
          >
       ];
       SubInput [
          shape=plaintext
          label=<
            <table border="0">
              <tr>
                <td border="1px" width="120" height="40" port="port">SubInput</td>
                <td border="0" width="140" height="40"><b>ForwardsSamples</b></td>
                <td border="0" width="120" height="40" align="left">direct forward</td>
              </tr>
            </table>
          >
       ];
       Input [
          shape=plaintext
          label=<
            <table border="0">
              <tr>
                <td border="1px" width="120" height="40" port="port">Input</td>
                <td border="0" width="140" height="40"><b>ForwardsSamples</b></td>
                <td border="0" width="120" height="40" align="left">broadcast matching (combine samples in cardinality)</td>
              </tr>
            </table>
          >
       ];
       InputGroup [
          shape=plaintext
          label=<
            <table border="0">
              <tr>
                <td border="1px" width="120" height="40" port="port">InputGroup</td>
                <td border="0" width="140" height="40"><b>ForwardsSamples</b></td>
                <td border="0" width="120" height="40" align="left">broadcast matching (combine samples in payload)</td>
              </tr>
            </table>
          >
       ];
       NodeC [
          shape=plaintext
          label=<
            <table border="0">
              <tr>
                <td border="1px" width="120" height="40" port="port">NodeRun</td>
                <td border="0" width="140" height="40"><b>ForwardsSamples</b></td>
                <td border="0" width="120" height="40" align="left">combines payloads (plugin based, e.g. cross product)</td>
              </tr>
            </table>
          >
       ];

       Output:port -> SubOutput:port [weight=25];
       Output:port -> Link:port [weight=10];
       SubOutput:port -> SubOutput:port [weight=0];
       SubOutput:port -> Link:port [weight=25];
       Link:port -> SubInput:port;
       SubInput:port -> Input:port;
       Input:port -> InputGroup:port;
       InputGroup:port -> NodeC:port;
    }

The idea is that we make a common interface for all classes that are related
to the flow of Samples. For this we propose the following mixin classes that
provide the interface and allow for better code sharing. The basic structure
of the classes is given in the following diagram:

.. graphviz::

    digraph mixins {
         node [
            fontname = "Bitstream Vera Sans"
            fontsize = 9
            shape = "record"
        ]

        edge [
            arrowtail = "empty"
        ]

        HasDimensions [
            shape = record
            label = "{HasDimensions|dimensions|+ size\l+ dimnames\l}"
        ];
        HasSamples [
            shape = record
            label = "{HasSamples|__getitem__()|+ __contains__\l+ __iter__\l+ iteritems()\l+ items()\l+ indexes\l+ ids \l}"
        ];
        ContainsSamples [
            shape = record
            label = "{ContainsSamples|samples|+ __getitem__()\l+ __setitem__()\l+ dimensions\l}"
        ];
        ForwardsSamples [
            shape = record
            label = "{ForwardsSamples|source\lindex_to_target\lindex_to_source\lcombine_samples\lcombine_dimensions|+ __getitem__\l+ dimensions\l}"
        ];

        HasDimensions -> HasSamples [dir=back];
        HasSamples -> ContainsSamples [dir=back];
        HasSamples -> ForwardsSamples [dir=back];
    }

The abstract and mixin methods are as follows:

=================== ================= ======================== ===================
ABC                 Inherits from     Abstract Methods         Mixin methods
=================== ================= ======================== ===================
``HasDimensions``                     | ``dimensions``         | ``size``
                                                               | ``dimnames``
``HasSamples``      ``HasDimensions`` | ``__getitem__``        | ``__contains__``
                                                               | ``__iter__``
                                                               | ``iteritems``
                                                               | ``items``
                                                               | ``indexes``
                                                               | ``ids``
``ContainsSamples`` ``HasSamples``    | ``samples``            | ``__getitem__``
                                                               | ``__setitem__``
                                                               | ``dimensions``
``ForwardsSamples`` ``HasSamples``    | ``source``             | ``__getitem__``
                                      | ``index_to_target``    | ``dimensions``
                                      | ``index_to_source``
                                      | ``combine_samples``
                                      | ``combine_dimensions``
=================== ================= ======================== ===================

.. note::
    Though the flow is currently working like this, the mixins are not yet created.

Network Execution
-----------------

The network execution should contain a number of steps:

* ``Network``

  * Creates a ``NetworkRun`` based on the current layout

* ``NetworkRun``

  * Transform the ``Network`` (possibly joining Nodes of certain interface into a combined NodeRun etc)
  * Start generation of the Job Direct Acyclic Graph (DAG)

* ``SchedulingPlugin``

  * Prioritize Jobs based on some predefined rules
  * Combine certain ``Jobs`` to improve efficiency (e.g. minimize i/o on a grid)

* ``ExecutionPlugin``

  * Run a (list of) ``Jobs``. If there is more than one jobs, run them sequentially on
    same execution host using a local temp for intermediate files.
  * On finished callback: Updated DAG with newly ready jobs, or remove cancelled jobs

This could be visualized as the following loop:

.. graphviz::

    digraph execution {
         node [
            fontname = "Bitstream Vera Sans"
            fontsize = 11
            shape = "box"
        ]

        Network;
        NetworkRun;
        NodeRun;
        JobDAG;
        SchedulingPlugin;
        ExecutionPlugin;

        Network -> NetworkRun [label=creates];
        NetworkRun -> JobDAG [label=creates];
        NetworkRun -> NodeRun [label=executes];
        NodeRun -> JobDAG [label="adds jobs"];
        JobDAG -> SchedulingPlugin [label="analyzes and selects jobs"];
        SchedulingPlugin -> ExecutionPlugin [label="(list of) Jobs to execute"];
        ExecutionPlugin -> NetworkRun [label=callback];
    }

The callback of the ``ExecutionPlugin`` to the ``NetworkRun`` would trigger
the execution of the relevant ``NodeRuns`` and the addition of more ``Jobs``
to the ``JobDAG``.

.. note:: The Job DAG should be thread-safe as it could be both read and
          extended at the same time.

.. note:: If a list of jobs is send to the ``ExecutionPlugin`` to be run as
          on Job on an external execution platform, the resources should be
          combined as follows: memory=max, cores=max, runtime=sum

.. note:: If there are execution hosts that have mutliple cores the
          ``ExecutionPlugin`` should manage this (for example by using pilot
          jobs). The ``SchedulingPlugin`` creates units that should be run
          sequentially on the resources noted and will not attempt
          parallelization

A ``NetworkRun`` would be contain similar information as the ``Network`` but 
not have functionality for editting/changing it. It would contain the
functionality to execute the Network and track the status and samples. This
would allow ``Network.execute`` to create multiple concurent runs that operate
indepent of each other. Also editting a ``Network`` after the run started would
have no effect on that run.

.. note:: This is a plan, not yet implemented

.. note:: For this to work, it would be important for a Jobs to have forward
          and backward dependency links.

SchedulingPlugins
~~~~~~~~~~~~~~~~~

The idea of the plugin is that it would give a priority on Jobs created by a
``Network``. This could be done based on different strategies:

* Based on (sorted) sample id's, so that one sample is always prioritized over
  others. The idea is that samples are process as much as possible in order,
  finishing the first sample first. Only processing other samples if there is
  left-over capacity.
* Based on distance to a (particular) ``Sink``. This is to generate specific
  results as quick as possible. It would not focus on specific samples, but
  give priority to whatever sample is closest to being finished.
* Based on the distance to from a ``Souce``. Based on the sign of the weight
  it would either keep all samples on the same stage as much as possible, only
  progressing to a new ``NodeRun`` when all samples are done with the previous
  ``NodeRun``, or it would push samples with accelerated rates.

Additionally it will group ``Jobs`` to be executed on a single host. This could
reduce i/o and limited the number of jobs an external scheduler has to track.

.. note::
    The interface for such a plugin has not yet been established.


Secrets
--------------------
"Something that is kept or meant to be kept unknown or unseen by others."

Using secrets
~~~~~~~~~~~~~~~~~~~~
Fastr IOPlugins that need authentication data should use the Fastr SecretService for retrieving such data. The SecretService can be used as follows.

.. code-block:: python

  from fastr.utils.secrets import SecretService
  from fastr.utils.secrets.exceptions import CouldNotRetrieveCredentials

  secret_service = SecretService()

  try:
    password = secret_service.find_password_for_user('testserver.lan:9000', 'john-doe')
  except CouldNotRetrieveCredentials:
    # the password was not found
    pass


Implementing a SecretProvider
~~~~~~~~~~~~~~~~~~~~
A SecretProvider is implemented as follows:

1. Create a file in fastr/utils/secrets/providers/<yourprovidername>.py
2. Use the template below to write your SecretProvider
3. Add the secret provider to fastr/utils/secrets/providers/__init__.py
4. Add the secret provider to fastr/utils/secrets/secretservice.py: import it and add it to the array in function _init_providers

.. code-block:: python

  from fastr.utils.secrets.secretprovider import SecretProvider
  from fastr.utils.secrets.exceptions import CouldNotRetrieveCredentials, CouldNotSetCredentials, CouldNotDeleteCredentials, NotImplemented


  try:
    # this is where libraries can be imported 
    # we don't want fastr to crash if a specific
    # library is unavailable
    # import my-libary
  except (ImportError, ValueError) as e:
    pass

  class KeyringProvider(SecretProvider):
    def __init__(self):
      # if libraries are imported in the code above
      # we need to check if import was succesfull
      # if it was not, raise a RuntimeError
      # so that FASTR ignores this SecretProvider
      # if 'my-library' not in globals():
      #   raise RuntimeError("my-library module required")
      pass

    def get_password_for_user(self, machine, username):
      # This function should return the password as a string
      # or raise a CouldNotRetrieveCredentials error if the password
      # is not found.
      # In the event that this function is unsupported a
      # NotImplemented exception should be thrown
      raise NotImplemented()

    def set_password_for_user(self, machine, username, password):
      # This function should set the password for a specified
      # machine + user. If anything goes wrong while setting
      # the password a CouldNotSetCredentials error should be raised.
      # In the event that this function is unsupported a
      # NotImplemented exception should be thrown
      raise NotImplemented()

    def del_password_for_user(self, machine, username):
      # This function should delete the password for a specified
      # machine + user. If anything goes wrong while setting
      # the password a CouldNotDeleteCredentials error should be raised.
      # In the event that this function is unsupported a
      # NotImplemented exception should be thrown
      raise NotImplemented()
