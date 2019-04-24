Fastr User Reference
====================

.. attribute:: fastr.tools

    A ToolManager containing all versions of all Tools loaded into the FASTR
    environment. The ToolManager can be indexed using the Tool id string or
    a tool id string and a version. For example if you have two versions (4.5
    and 4.8) of a tool called *Elastix*:

    .. code-block:: python

        >>> fastr.tools['elastix.Elastix']
        Tool Elastix v4.8 (Elastix Registration)
                                   Inputs                              |             Outputs
        --------------------------------------------------------------------------------------------------
        fixed_image       (ITKImageFile)                               |  directory (Directory)
        moving_image      (ITKImageFile)                               |  transform (ElastixTransformFile)
        parameters        (ElastixParameterFile)                       |  log_file  (ElastixLogFile)
        fixed_mask        (ITKImageFile)                               |
        moving_mask       (ITKImageFile)                               |
        initial_transform (ElastixTransformFile)                       |
        priority          (__Elastix_4.8_interface__priority__Enum__)  |
        threads           (Int)                                        |
    
        >>> fastr.tools['elastix.Elastix', '4.5']
        Tool Elastix v4.5 (Elastix Registration)
                                   Inputs                              |             Outputs
        --------------------------------------------------------------------------------------------------
        fixed_image       (ITKImageFile)                               |  directory (Directory)
        moving_image      (ITKImageFile)                               |  transform (ElastixTransformFile)
        parameters        (ElastixParameterFile)                       |  log_file  (ElastixLogFile)
        fixed_mask        (ITKImageFile)                               |
        moving_mask       (ITKImageFile)                               |
        initial_transform (ElastixTransformFile)                       |
        priority          (__Elastix_4.5_interface__priority__Enum__)  |
        threads           (Int)                                        |

.. attribute:: fastr.types

    A dictionary containing all types loaded into the FASTR environment. The keys are the typenames and the values are the classes.

.. attribute:: fastr.networks

    A dictionary containing all networks loaded in fastr

.. automethod:: fastr.api.create_network

.. automethod:: fastr.api.create_network_copy

.. autoclass:: fastr.api.Network
    :members: id, version, create_node, create_link, create_sink, create_source,
              create_constant, create_macro, draw, execute, load, save, nodes

.. autoclass:: fastr.api.Link
    :members: id, collapse, expand

.. autoclass:: fastr.api.Node
    :members: id, inputs, outputs, input, output

.. autoclass:: fastr.api.Input
    :members: id, input_group, append, __lshift__, __rrshift__

.. autoclass:: fastr.api.Output
    :members: id, __getitem__

