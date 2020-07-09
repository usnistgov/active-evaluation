import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import os
import logging


class ExperimentParams:
    """
    Python class to store experimental parameters, akin to a struct that stores variables. Here
    are the variables that are stored:

    self.num_step_samples (int):
        the number of samples to take at each round
    self.num_success_rounds_required (int):
        the number of successful rounds
    self.alpha (num):
        the alpha (1 - probability) value
    self.delta (num):
        the delta number of the range of uncertainty
    self.num_strata (int):
        the number of strata
    self.stratification_type (`aegis.acteval.strata.Strata`):
        the strata class object specifying the stratification strategy
    self.bin_style (str):
        The style to stratify the bins. Default is 'equal'. Values are:

        'equal'
            Stratify the bins so that the range of values is equal, or the bins
            are of equal width.
        'perc'
            Stratify by percentile, or so that an equal number of
            trials are in each bin.
    self.metric_obj (`aegis.acteval.metrics.Metric`):
        The reference to the metric object that specifies how the trials will be scored
    sampler_type (`aegis.acteval.samplers.TrialSampler`):
        The type of sampler specifying sampling strategy
    request_initial_samples (bool):
        A boolean that determines if the experiment (and hence) the
        controller should request or supplement initial samples with initial samples to provide
        for adequate stratum and metric coverage for initial estimates. True by default
    initial_samples (int):
        The number of samples total divided evenly between "bins" to
        request initially. The default value is 50 samples per bin. It is set this high in order
        that we have enough samples for approximate CI estimates to be reasonable.
    """

    def __init__(self, num_step_samples=100, num_success_rounds_required=2,
                 alpha=0.05, delta=0.01, num_strata=4,
                 stratification_type=aegis.acteval.strata.StrataFirstSystem,
                 bin_style="perc", metric_object=aegis.acteval.metrics.BinaryAccuracyMetric(),
                 sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                 request_initial_samples=True, initial_samples=200):
        """
        Constructor.

        Args:
            num_step_samples (int, optional): the number of samples to ask for at each iteration;
                defaults to 100
            alpha (num, optional): The specified probability \\alpha. Defaults to 0.05.
            delta (num, optional): The specified interval range \\delta. Defaults to 0.01
            num_success_rounds_required (int, optional):
                The number of rounds where the (1-\\alpha) confidence
                interval's range is within +- $\\delta width. Defaults to 2
            num_strata (int, optional): The number of strata to have. Defaults to 4
            stratification_type (:obj:`aegis.acteval.strata.Strata`, optional):
                Strata class that gives the stratification strategy and type of strata
            metric_object (:obj:`aegis.acteval.metrics.Metric`, optional):
                The reference to the metric object that specifies how the trials will be scored
            sampler_type (:obj:`aegis.acteval.samplers.TrialSampler`, optional):
                The type of sampler specifying sampling strategy
            bin_style (str, optional):
                The style to stratify the bins. Default is 'equal'. Values are:

                'equal'
                    Stratify the bins so that the range of values is equal, or the bins
                    are of equal width.
                'perc'
                    Stratify by percentile, or so that an equal number of
                    trials are in each bin.
            request_initial_samples (bool):
                A boolean that determines if the experiment (and hence) the
                controller should request or supplement initial samples with initial samples
                to provide for adequate stratum and metric coverage for initial estimates.
                True by default
            initial_samples (int): The number of samples to request initially.
        """
        self.num_step_samples = num_step_samples
        self.num_success_rounds_required = num_success_rounds_required
        self.alpha = alpha
        self.delta = delta
        self.num_strata = num_strata
        self.stratification_type = stratification_type
        self.bin_style = bin_style
        self.metric_object = metric_object
        self.sampler_type = sampler_type
        self.request_initial_samples = request_initial_samples
        self.initial_samples = initial_samples
        logger = logging.getLogger("paper_experiments_logger"+"." + str(stratification_type) +
                                   "." + str(sampler_type) + "." + str(bin_style))
        logger.setLevel(logging.DEBUG)
        for hdlr in logger.handlers[:]:  # remove all old handlers
            logger.removeHandler(hdlr)
        logfile_fpath = os.path.join(str(stratification_type) + "." + str(sampler_type) + "." +
                                     str(bin_style) + '.tmp')
        fh = logging.FileHandler(logfile_fpath)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.info("Experiment"+" " + str(stratification_type) +
                    " " + str(sampler_type) + " " + str(bin_style))

    def __str__(self):
        """
        __str__ method to Display experimental parameters.

        Returns:
            str: a printable string of experiment parameters.
        """

        request_initial_samples_str = "Did not request initial samples for initial coverage."
        if self.request_initial_samples:
            request_initial_samples_str = "Did request initial samples for initial coverage."

        return ("Experimental Parameters:\n\t" +
                str(self.num_step_samples) + " samples per round with " +
                str(self.num_success_rounds_required) + " successful rounds required, alpha=" +
                str(self.alpha) + ", delta=" + str(self.delta) + ".\n\tTakes " +
                str(self.num_strata) + " strata with stratification type " +
                str(self.stratification_type) + " using bin style " + str(self.bin_style) +
                ".\n\tUses metric object " + str(self.metric_object) + ".\n\tUses sampler type " +
                str(self.sampler_type) + ".\n\t" +
                str(request_initial_samples_str) + " Requested " +
                str(self.initial_samples) + " samples requested.")

    def get_experiment_tuple(self):
        """
        Create a tuple of experiment variables and return them. The ordering of the tuples is
        important, since this tuple will be the basis for a data frame row

        Returns:
            tuple: (int, int, num, num, aegis.acteval.strata.Strata, str,
            aegis.acteval.metrics.Metric, aegis.acteval.samplers.TrialSampler,
            bool, int): A tuple of all of the stored experimental value. See the class
            documentation for a description of each of these variables.

        """
        return (self.num_step_samples,
                self.num_success_rounds_required,
                self.alpha,
                self.delta,
                self.num_strata,
                self.stratification_type,
                self.bin_style,
                self.metric_object,
                self.sampler_type,
                self.request_initial_samples,
                self.initial_samples,)
