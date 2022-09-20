.. api-content documentation master file, created by
   sphinx-quickstart on Mon Jun 20 10:52:22 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ONNX static content generation
=======================================

This is a developer usage guide to the ONNX Python API and Operator Schemas. It contains the following information for the latest release:

** API Overview **
For all modules involved in the different tools part of the API, this usage guide pulls the functions and defines them. All information is auto-generated and will update every time the docs are re-built.

** Operators and Op Schemas **
Lists out all the ONNX operators. For each operator, lists out the usage guide, parameters, examples, and line-by-line version history.
This section also includes tables detailing each operator with its versions, as done in Operators.md.


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Overview
   
   onnx-api/modules/*


.. toctree::
	:glob:
	:maxdepth: 1
	:caption: Operators + OpSchemas

	include

.. toctree::
	:glob:
	:maxdepth: 1
	:caption: Version History

.. toctree::
	:glob:
	:maxdepth: 1
	:caption: Notes