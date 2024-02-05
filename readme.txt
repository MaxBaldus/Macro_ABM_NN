Implementing and simulating the base macroeconomic agent based model by Delli Gatti et al. (2011), referred as BAM model, as well as the toy macro abm by Caiani et al. (2016). 
The BAM model ist estimated using US and German GDP and combining the approaches by Platt (2022) and Delli Gatti & Grazzini (2020): Bayesian estimation is exploited and the likelihood function is approximated via mixture densitiy networks (MDN). Latin hypercube sampling is used to set up the grid search of the parameter space.

________________________________________

Delli Gatti, D., Desiderio, S., Gaffeo, E., Cirillo, P., & Gallegati, M. (2011). Macroeconomics from the bottom-up (Vol. 1). Springer Science & Business Media.
(New Economic Windows) Alessandro Caiani, Alberto Russo, Antonio Palestrini, Mauro Gallegati (eds.) - Economics with Heterogeneous Interacting Agents_ A Practical Guide to Agent-Based Modeling, Chapter 2
Platt, D. (2022). Bayesian estimation of economic simulation models using neural networks. Computational Economics, 59(2), 599-650.
Delli Gatti, D., & Grazzini, J. (2020). Rising to the challenge: Bayesian estimation and forecasting techniques for macroeconomic agent based models. Journal of Economic Behavior & Organization , 178 (2020), 875902.

Estimation code is tailored to the BAM model and grid search approach, using the package:

Approximate Likelihood Estimation using Neural Networks (ALENN)
Donovan Platt
Mathematical Institute, University of Oxford
Institute for New Economic Thinking at the Oxford Martin School
Copyright (c) 2020, University of Oxford. All rights reserved.


python version used: 3.10.6
