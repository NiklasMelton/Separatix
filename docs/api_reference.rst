API reference
=============

The reference below is generated directly from the public Python objects and
their source docstrings.

Functional API
--------------

.. autofunction:: separatix.diagnose

Estimator API
-------------

.. autoclass:: separatix.ComplexityProfiler
   :members: fit, report, recommendation
   :show-inheritance:

Configuration
-------------

.. autoclass:: separatix.ProfilerConfig
   :members: to_dict

Report object
-------------

.. autoclass:: separatix.DiagnosticReport
   :members: to_dict, to_json
