import pandas as pd


class System(object):
    """
    Class to representing a system.

    system_id the name or id of the system as a string

    system_df is a data frame that
            has a column trial_id, followed by one column headed with field system_id.
            Each value is the score of that trial for that system.

    threshold_df a three-column data-frame with columns (system_id, threshold, value) that
            encodes threshold information. This supports multiple thresholds that can be named
            when necessary.
    """

    # Computed values will be stored in the system objects rather
    population = 0
    population_variance = 0
    population_frac = 0
    population_frac_variance = 0
    score = 0
    score_lower = 0
    score_upper = 0
    score_variance = 0
    score_variance_upper = 0
    sampled_trials = 0
    confidence_value = 0

    def __init__(self, system_id, system_df, sys_threshold_df):
        """
        Constructor for a System object.

        Args:
            system_id (str): the name or id of the system as a string
            system_df (:obj:`pandas.core.frame.DataFrame`): is a data frame that
                has a column trial_id, followed by one column headed with field system_id.
                Each value is the score of that trial for that system.
            sys_threshold_df (:obj:`pandas.core.frame.DataFrame`): a three-column data-frame
                with columns (system_id, threshold, value)
                that encodes threshold information. This supports multiple thresholds
                that can be named when necessary.
        """
        self.system_id = system_id
        self.system_df = system_df
        self.threshold_df = sys_threshold_df

    def __str__(self):
        """
        __str__ method to reproduce a string of the System object

        Returns:
            str: the string of the system information for printing
        """
        return ("System " + str(self.system_id) + "\n\tScore: " +
                str(self.score) + " +/- " + str(self.confidence_value) +
                "\n\tScore Variance (standard error squared): " + str(self.score_variance) +
                "\n\tNumber of counted sampled trials: " + str(self.sampled_trials) +
                " out of " + str(self.population) + " countable trials.")

    def add_stratum_index_to_system_data(self, stratum_ind_df):
        """
        Adds the stratum_index column to the data frame.

        Args:
            stratum_ind_df (:obj:`pandas.core.frame.DataFrame`): The data frame with columns
                "trial_id" and "stratum_index"

        Returns:
            pandas.core.frame.DataFrame: sys_df a reference to the modified systems data frame

        """
        # TODO: Should match. Report an error if the merge does not match
        self.system_df = pd.merge(self.system_df, stratum_ind_df, how="inner", on="trial_id")
        return self.system_df

    def get_system_tuple(self):
        """
        Returns a tuple of system information, useful to convert into a pandas data frame.

        Returns:
            tuple: a tuple of system information that can be converted into a data frame.

        """
        return (self.system_id, self.score, self.confidence_value, self.score_variance,
                self.sampled_trials, self.population)
