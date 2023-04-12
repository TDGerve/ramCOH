.. include:: ./substitutions.rst

=========
Changelog
=========

v1.1.2
------
Bugfix for co2.py not being recognised due to capitalisation in filename.

v1.1.1
------
Minor bugfixes for compatibility issues with `SilicH2O <https://silich2o.readthedocs.io/en/latest/>`_ .

v1.1.0
------
* The :py:mod:`~ramCOH.raman.glass.Glass` module has been added for processing Raman spectra of silicate glasses and quantifying their water contents.
* The :py:meth:`~ramCOH.raman.co2.CO2.FermiDiad` method of the :py:class:`~ramCOH.raman.co2.CO2` class has been update to use peak deconvolution instead of simple peak fitting.
  