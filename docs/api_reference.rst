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

Probe recipes and audit factory
-------------------------------

.. autoclass:: separatix.ProbeRecipe
   :members: from_dict, from_json, from_estimator, recipe_id, schema, schema_version, probe, implementation, estimator_spec, to_dict, as_dict, to_json

.. autofunction:: separatix.build_probe_recipe

.. autofunction:: separatix.make_probe_estimator

Recipe exceptions
-----------------

.. autoexception:: separatix.ProbeRecipeError

.. autoexception:: separatix.UnsupportedProbeRecipeVersion

.. autoexception:: separatix.ProbeRecipeCompatibilityError
