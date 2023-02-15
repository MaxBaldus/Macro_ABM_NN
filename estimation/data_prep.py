import statsmodels.api as sm
import matplotlib.pyplot as plt

""" In the following class all filters used throughout simulations and estiamtions are defined."""

class Filters:

    def __init__(self, GDP, inflation, unemployment):
        
        """ time series to be filtered """
        self.gdp = GDP 
        # self.inflation = inflation
        # self.unemployment = unemployment
    
    """For the real GDP data (from FRED) and simlated GDP time series the HP-filer is set-up with an lambda value of 1600 (for quartely data)
    It is applied to the real GDP data (from FRED) and simlated GDP time series to extract the cyclical component."""

    def HP_filter(self, empirical:bool):
        
        cycle, trend = sm.tsa.filters.hpfilter(self.gdp, 1600) # HP filter for the cycle and trend of the ts x with smoothing value lambda of 1600
        # components = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1]})
        
        if empirical:
            plt.figure("GDP Components") # log GDP figure
            plt.plot(self.gdp, label = "GDP") # GDP 
            plt.plot(cycle, label = "cycle") # cycle
            plt.plot(trend, label = "trend") # trend
            plt.xlabel("Time")
            plt.ylabel("Log output")
            plt.legend()

            plt.show()

        return cycle, trend

        
    
    # filters for inflation and unemployment
    


        


        


