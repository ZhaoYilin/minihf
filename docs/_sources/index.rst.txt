.. minihf documentation master file, created by
   sphinx-quickstart on Tue May 19 10:40:33 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================
MiniHF: A Miniimal Hartree-Fock Library
=======================================

MiniHF is a minimal, educational Hartree-Fock quantum chemistry library 
written in Python. It provides a clean implementation of molecular integral 
calculations and self-consistent field methods for electronic structure theory.

Key Features
------------

- **Molecular Integrals**: Efficient computation of overlap, kinetic energy, 
  nuclear attraction, and electron repulsion integrals using the Obara-Saika 
  recurrence scheme.

- **Hartree-Fock Methods**: Implementation of Restricted Hartree-Fock (RHF), 
  Unrestricted Hartree-Fock (UHF), and Restricted Open-shell Hartree-Fock (ROHF).

- **Basis Sets**: Support for standard Gaussian basis sets in NWChem format.

- **Educational Focus**: Clean, well-documented code designed for learning 
  quantum chemistry algorithms.

- **Lightweight**: Minimal dependencies, easy to install and use.

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Theory & Tutorials

   integral
   hf

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
-------

MiniHF is released under the MIT License. See the LICENSE file for details.

Contributing
------------

Contributions are welcome! Please see the GitHub repository for guidelines.

GitHub Repository: https://github.com/yilinzhang/minihf