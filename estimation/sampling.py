"""
Here two sampling algorithms are implemented which sample the posterior distribution and output the approximated densities.
The first simply uses a pre-defined grid with equally spaced parameter values.
The second one is a metropolis hastings algorithm which explores the parameter space (computationally more evolved).
"""

# libraries

# classes
from mde import mdn


class sample_posterior:

    def __init__(self, model):
        
        """
        Here the model class is loaded s.t. for each parameter value the likelihood is apprixmated using the specified mdn. 
        """

        self.model = model
    
    def grid_search(self):

        curr_model = self.model()
        # construct equally spaced grid for each parameter 

        # for theta_i in values:

            # create model instance
            # simulate data with theta[i]
            # do mdn (likelihood block) ll
            # compute prior proba p(theta)
            
            # return ll + p(theta) and ll

    def MH_sampling(self):

        print("blub")
        # now mh sampling
        # 

    def posterior_plots(self):

        print("blub")
        # output posterior estimates (mean of theta chain)
        # plot the sampled posterios 