Operator categories
===================

Chainable accessors reached on a waveform as ``wave.filter``, ``wave.amplitude``,
and ``wave.math``. Length-preserving methods return a new lazy
:class:`~wave_measure.Waveform`; terminal reductions return a result.

.. autoclass:: wave_measure.categories.FilterCategory
   :members:

.. autoclass:: wave_measure.categories.AmplitudeCategory
   :members:

.. autoclass:: wave_measure.categories.MathCategory
   :members:
