DISTRIBUTIONS = dict(
        norm = 'Normal',
        expon = 'Exponential',
        lognorm = 'Lognormal',
        gamma = 'Gamma',
        foldnorm = 'Folded Normal',
        foldcauchy = 'Folded Cauchy',
        exponpow = 'Exponential Power',
        gengamma = 'Generalised Gamma',
        gennorm = 'Generalised Normal',
        genlogistic = 'Generalised Logistics',
        exponweib = 'Exponentiated Weibull',
        weibull_max = 'Weibull Maximum',
        weibull_min = 'Weibull Minimum',
        pareto = 'Pareto',
        genextreme = 'Generalized Extreme Value',
        gausshyper = 'Gauss Hypergeometric',
        hypsecant = 'Hyperbolic Secant',
        invgamma = 'Inverted Gamma',
        laplace = 'Laplace ',
        loglaplace = 'log-Laplace',
    )
DISTRIBUTION_OPTIONS = [{'label': str(DISTRIBUTIONS[distribution]),
                         'value': str(distribution)}
                         for distribution in DISTRIBUTIONS]

DESCRIPTION = """
When an analyst wants to simulate a queueing process in a F&B outlet or the length of stay of patients in a particular specialty, \
we need to verify which distribution can be used to generate simulated data based on the actual data. We often utilise \
Chi-Square Test to do so. In this simple application, we identify the best distribution with the steps below:
"""