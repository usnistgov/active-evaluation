class SummaryReport:
    """
    Summary Report Object that stores the experimental conditions and results in an object
    with an easy way to print the results to any output stream. The constructor takes in the
    summary report values and stores them into this object, and the class overrides the
    __repr__() method to allow for easy printing, and has the __str__ method call __repr__() for
    ease of understanding.
    """

    def __init__(self, experiment, system_list, num_rounds, num_init_trials, num_total_samples,
                 num_requested_init_trials, num_stratum_at_end, num_samples_per_stratum):
        """
        Constructor that takes in experimental parameters and experimental outputs
        and then stores them in this class.

        Args:
            experiment(:obj:`aegis.acteval.experiment.ExperimentParams`):
                The Experiment Object with all of the experimental parameters.
            system_list(list of :obj:`aegis.acteval.system.System`):
                The list of System objects
            num_rounds (int): the number of rounds taken in the experiment
            num_init_trials (int): the number of initial trials provided
            num_total_samples (int): the total number of samples taken. This can differ from
                :math:`num_rounds*experiment.num_step_samples + num_init_trials` if a stratum
                is exhausted and some or no samples are taken.
            num_requested_init_trials (int): the total number of additional initial samples
                generated. This is 0 if initial samples were not requested
            num_stratum_at_end (int): The number of nonempty stratum produced.
            num_samples_per_stratum (list of int): a list of the amount of samples
                in each stratum
        """
        self.experiment = experiment
        self.system_list = system_list
        self.num_rounds = num_rounds
        self.num_init_trials = num_init_trials
        self.total_sampled_trials = num_total_samples
        self.num_requested_init_trials = num_requested_init_trials
        self.num_stratum_at_end = num_stratum_at_end
        self.num_samples_per_stratum = num_samples_per_stratum

    def __str__(self):
        """
        __str__ method used to provide pretty printing.

        Returns:
            str: A pretty-printed string of the summary report with useful experiment information.

        """
        output_str = str(self.experiment) + '\nTotal number of rounds: ' + str(self.num_rounds) + \
            ", requiring a total sample of " + \
            str(self.total_sampled_trials) + " trials.\n" + "Requested " + \
            str(self.num_requested_init_trials) + " additional initial trials, " + \
            "In addition to the " + \
            str(self.num_init_trials) + " trials provided by init_df.\n" + \
            "Ended with " + str(self.num_stratum_at_end) + " non-empty stratum."
        for system in self.system_list:
            output_str += str(system) + "\n"
        return output_str

    def get_experiment_report_tuple(self):
        """
        Takes the variables and produces a report tuple. System results are not present.
        This method is combined with :func:`get_systems_tuples()` to get system reports.

        Returns:
            tuple: A tuple of all of the variables in the report

        """
        exp_tuple = self.experiment.get_experiment_tuple()
        summary_tuple = (self.total_sampled_trials,
                         self.num_rounds,
                         self.num_init_trials,
                         self.num_requested_init_trials,
                         self.num_stratum_at_end)
        return exp_tuple + summary_tuple

    def get_systems_tuples(self):
        """
        Returns a list of tuples of system outputs, one tuple per system.

        Returns:
            list of tuple: a list of experiment report tuples, one tuple per system.

        """
        return [system.get_system_tuple() for system in self.system_list]
