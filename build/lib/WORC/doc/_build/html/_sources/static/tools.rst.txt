Guidlines on using and creating Fastr tools
===========================================


Fastr pulls its tools definitions from a repository called FastrHub. Optionally tool definitions can also be pulled from a
local folder or a private repository.

.. figure:: images/tools/tooldefresolveflowchart.png
   :alt: A flowchart of the Fastr tool loading order. Priority is as follows: 1 load from cache, 2 load from local disk, 3 load from FastrHub repository, 4 raise Fastr Cannot Find Tool Definition Exception.

   Fastr tool definition loading order.


Creating a tool definition
--------------------------

The tool definition (xml file) is a wrapper. We should version these wrappers. Use the version property in the tool element to do this.


Uploading a tool definition to FastrHub
---------------------------------------

Tool definitions in FastrHub can not be overwritten. This is to improve reproducability. A tool definition can be changed to 'deprecated' which hides it from search but it will still be downloadable.
The example below pushes the FSLMaths tool into the fsl namespace. It uses definition v1.0. This is the version property in the tool-element of your tool-wrapper xml.

.. code-block::

   $ fastr tools push fsl/FSLMaths 1.0
     Username: demo
     Password: *****
     Uploading [########--] 80%



Code examples
-------------
Instantiate a node that uses the FSLMaths tool.

.. code-block:: python

   # network.create_node('<required:namespace>/<required:tool>:<required:command_version>', tool_version='<required:tool_version>', repositoy='<optional:repo url, defaults to FastrHub>')
   node = network.create_node('fsl/FSLMaths:5.0.9', tool_version='1.0')
