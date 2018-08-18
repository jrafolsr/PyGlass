# PyGlass
Module to simulate the specific heat signature of glasses with a specified thermal treatment following the Tool-Narayanaswamy-Moynihan (TNM) model

The class ```Glass``` creates a glass object for which you can define a thermal history through the different thermal treatments: ramp (either heating or cooling) and annealing. After defining its history, you can then compute the fictive temperature and normalized heat capacity through the TNM model, with an evolution for the relaxation time that can follow the traditional tnm model or the Adam-Gibbs-Vogel model. More to be implemented.

The Notebook ``Example_usage_PyGlass_module.ipynb`` shows a complete example of hw to use the ``Glass`` class.
