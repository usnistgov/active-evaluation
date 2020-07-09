import pandas as pd
import numpy as np


def str_dec(x):
    """
    Utility function used by stratum method to avoid redefinition with lambda

    Args:
        x (object):  the object to make a string and to construct a string with

    Returns:
        str: a string with "_dec" appended to it

    """
    return str(x) + '_dec'


class Stratum(object):
    """
    An object representing a stratum.

    In the stratum, the population size
    is the total number of trials in the stratum, not the total number of trials that have been
    sampled. The number of samples (sampled_trials) is the total number of trials sampled.
    For metrics such as recall where this cannot be computed directly, this will be
    estimated by the total number of trials in the stratum and the proportion of sampled trials
    that should be included in the population.
    """

    def __init__(self, system_list):
        """
        Constructor.

        Args:
            system_list (list of :obj:`aegis.acteval.system.System`): The list of systems
                with outputs to associate with this stratum.
        """
        self.num_trials = 0
        # field 'key' is NA when there is no key, indicating that the trial was not sampled
        self.stratum_key_df = pd.DataFrame()
        self.system_list = system_list
        # -1 means unassigned index
        self.stratum_index = -1
        self.combined_df = pd.DataFrame()
        self.use_cached_combined_df = False
        self.score_df = pd.DataFrame()
        self.use_cached_score_df = False

    def dirty_stratum_cache(self):
        """
        Mark all cached variables as dirty for future computations within this stratum.

        Returns:
            Nothing

        """
        self.use_cached_score_df = False
        self.use_cached_combined_df = False
        return

    def get_combined_data_frames(self, system_data_frames, key_data_frame):
        """
        Internal implementation method to get combined data frames that can be used both
        to get the entire combined data frame and to get the data frame with only the key values
        that are not missing. Written so that the logic of combining data frames can be written
        once and called smartly.

        Args:
            system_data_frames (list of :obj:`pandas.core.frame.DataFrame`):
                The list of system data frames, pre-filtered as appropriate
            key_data_frame (:obj:`pandas.core.frame.DataFrame`):
                The key data frame, filtered as approporiate. Has structure
                with two columns (trial_id, key)

        Returns:
            pandas.core.frame.DataFrame: A combined data frame with all system scores,
            stratum index, key, and decision when applicable.

        """
        system_ids = [system.system_id for system in self.system_list]

        # Convert to one data frame with same data, but different outputs of score_1 ... score_k
        comb_df = system_data_frames[0].copy(deep=True)
        comb_df["system_id"] = system_ids[0]

        # Make sure comb_db isn't empty for appending, so filter each append and at the end
        # but do not filter here
        have_stratum_indices = comb_df.columns.isin(['stratum_index']).any()

        for i in range(1, len(system_data_frames)):
            temp_df = system_data_frames[i].copy(deep=True)
            temp_df["system_id"] = system_ids[i]
            # Filter before merging for faster computation
            if have_stratum_indices:
                temp_df = temp_df.loc[temp_df['stratum_index'] == self.stratum_index, :]
            comb_df = comb_df.append(temp_df, sort=True)
        # Filter before merging for faster computation
        if have_stratum_indices:
            comb_df = comb_df.loc[comb_df['stratum_index'] == self.stratum_index, :]

        output_score_df = comb_df.loc[:, ["trial_id", "system_id", "score"]]
        output_df = output_score_df.pivot(
            index="trial_id", columns="system_id", values="score"
        )
        output_df.reset_index(drop=False, inplace=True)

        # Add decision column if it exists with "<system_id>_dec" as the columns
        if 'decision' in system_data_frames[0]:
            output_dec_df = comb_df.loc[:, ["trial_id", "system_id", "decision"]]
            output_dec_df["system_id"] = \
                output_dec_df["system_id"].apply(str_dec)
            output_dec_df = output_dec_df.pivot(
                index="trial_id", columns="system_id", values="decision"
            )
            output_dec_df.reset_index(drop=False, inplace=True)
            output_df = output_df.merge(output_dec_df, how="left", on=["trial_id"])

        # merge with key data frame if not empty to get stratum indices and data
        if not key_data_frame.empty:
            output_df = output_df.merge(key_data_frame, how="left", on="trial_id")

        return output_df

    def get_combined_systems_df(self):
        """
        Takes the system data frames with system scores and combines to get one data frame
        that also includes the stratum indices, providing a combined data frame for this
        stratum.

        This method is cached in order to prevent redundant computations for performance
        enhancements.

        If the "stratum_index" field has already been added to the system data frames, then
        that field will be in the returned data frame. Otherwise, if the field "stratum_index"
        has not yet been added, then the "stratum_index" field will not appear in this combined
        data frame.

        Returns:
            pandas.core.frame.DataFrame: output_df, a copy of the combined data frame,
            where each system_id is a column whose value is the score.

        """
        if self.use_cached_combined_df:
            return self.combined_df

        system_data_frames = [system.system_df for system in self.system_list]
        key_data_frame = self.stratum_key_df
        output_df = self.get_combined_data_frames(system_data_frames, key_data_frame)
        self.combined_df = output_df
        self.use_cached_combined_df = True
        return output_df

    def get_combined_systems_score_df(self):
        """
        Takes the system data frames with system scores and combines to get one data frame
        that also includes the stratum indices, providing a combined data frame for this
        stratum. Gets the dataframe only for values that has keys.

        This allows for filtration before merging to save on computation.

        This method is cached in order to prevent redundant computations for performance
        enhancements.

        If the "stratum_index" field has already been added to the system data frames, then
        that field will be in the returned data frame. Otherwise, if the field "stratum_index"
        has not yet been added, then the "stratum_index" field will not appear in this combined
        data frame.

        Returns:
            pandas.core.frame.DataFrame: output_df a copy of the combined data frame, where each
            system_id is a column whose value is the score.

        """

        if self.use_cached_score_df:
            return self.score_df

        if self.stratum_key_df.empty:
            # need to figure out this case
            # return self.stratum_key_df
            pass

        key_data_frame = self.stratum_key_df.loc[pd.notna(self.stratum_key_df["key"]), :]
        if key_data_frame.empty:
            return key_data_frame

        system_data_frames = [system.system_df for system in self.system_list]
        # Here, stratum_index values should match exactly
        system_key_frames = [system_df.merge(key_data_frame, how="inner",
                                             on=["trial_id", "stratum_index"])
                             for system_df in system_data_frames]

        output_df = self.get_combined_data_frames(system_key_frames, key_data_frame)
        self.score_df = output_df
        self.use_cached_score_df = True
        return output_df

    def construct_stratum_from_trials(self, stratum_key_df, stratum_index):
        """
        A method to pass the information from the data frame to the stratum with the desired trials.

        Makes a deep copy so that the strata df is separated from the stratum df so that each can
        be modified separately when samples are added.

        Args:
            stratum_key_df (:obj:`pandas.core.frame.DataFrame`): The subset of trials from
                trial_df that should be copied to be the df in
                this strata
            stratum_index (int): The index of this stratum

        Returns:
            Nothing, since this updates the stratum object

        """
        self.stratum_index = stratum_index
        self.stratum_key_df = stratum_key_df.copy(deep=True)
        self.num_trials = stratum_key_df.shape[0]
        # may be incorrect if the metric differs, but use for now
        self.population_size = stratum_key_df.shape[0]
        self.sampled_trials = self.stratum_key_df.loc[
            pd.notna(stratum_key_df["key"]), "trial_id"
        ]
        pass

    def find_needed_initial_samples(self, metric, initial_samples_per_bin, rng):
        """
        In order to cover necessary edge cases from each stratum
        (Such as when the metric is a combination of two components) and to ensure samples
        from each stratum, this method will find such examples and will sample at random.

        Args:
            metric (:obj:`aegis.acteval.stratum.Stratum`): The metric to score systems
            initial_samples_per_bin (int): The number of initial samples per bin
            rng (:obj:`numpy.random.RandomState`): random state object with stored random state

        Returns:
            list of object: A list of trial_id values to sample

        """
        return metric.find_needed_initial_samples(self, initial_samples_per_bin, rng)

    def add_samples_to_stratum(self, samples_df):
        """
        Adds relevant scores to the trials in the current data frame that exist in the key field
        from the samples_df data frame.

        Args:
            samples_df (:obj:`pandas.core.frame.DataFrame`): the scored trials to add.
                If trials in samples_df are not trials
                already assigned to the stratum, they are ignored.

        Returns:
            aegis.acteval.stratum.Stratum: The updated stratum object
        """
        if samples_df is None:
            return
        if samples_df.shape[0] == 0:
            return

        # Merge data frame to combined df.
        self.stratum_key_df = self.stratum_key_df.merge(samples_df, how="left", on=["trial_id"])
        self.stratum_key_df.loc[
            pd.isna(self.stratum_key_df["key_x"]), "key_x"
        ] = self.stratum_key_df.loc[pd.isna(self.stratum_key_df["key_x"]), "key_y"]
        self.stratum_key_df.drop(["key_y"], axis=1, inplace=True)
        self.stratum_key_df.rename(columns={"key_x": "key"}, inplace=True)

        # Mark cached values as dirty when we add samples
        self.dirty_stratum_cache()
        return self

    def draw_samples(self, num_samples, metric, rng):
        """
        Draws relevant samples in each stratum. Uses the metric to make sure that it only draws
        relevant samples. (I.e. for precision, relevant samples or samples from the population are
        all systems where some systems' decision is 1).

        Args:
            num_samples (int): The number of samples to draw from this stratum
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object
            rng (:obj:`numpy.random.RandomState`): the random state object from the sampler to
                use for sampling

        Returns:
            list of object: a list of trial ids to sample

        """
        # sample_space = set(range(self.population_size)) - self.sampled_indxs
        # Call metric population
        unsampled_trials = metric.get_trials_to_sample_from(self)
        if num_samples > unsampled_trials.shape[0]:
            if unsampled_trials.shape[0] == 0:
                return []
            else:
                num_samples = unsampled_trials.shape[0]

        # Right now, this is unseeded. We may wish to have a seed either here or in the entire
        # method
        new_sample = unsampled_trials.sample(n=num_samples, replace=False, random_state=rng)
        return sorted(list(new_sample))

    def estimate_samples_all_systems(self, metric):
        """
        Estimates the number of samples (that count towards the population) for each system.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric estimating the
                number of samples

        Returns:
            list of int: A list with one entry per system of the number of samples in
            this stratum for that system.

        """
        return metric.estimate_samples_all_systems(self)

    def estimate_pop_all_systems(self, metric):
        """
        Estimates the population for each system for this stratum.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric estimating the population

        Returns:
            list of float: A list with one entry per system of the population of this stratum for
            that system. Whether the estimates are floats or ints depends on the metric.

        """
        return metric.estimate_pop_all_systems(self)

    def estimate_pop_upper_all_systems(self, metric, alpha):
        """
        Using the provided system aggregation strategy, estimates the upper bound of the population
        of this stratum.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric.
            alpha (float): the confidence value

        Returns:
            list of float: A list with one entry per system of the upper bound of the
            population of this stratum for that system.
            Whether the estimates are floats or ints depends on the metric.

        """
        sys_pop_conf_lists = metric.estimate_population_intervals_all_systems(self, alpha)
        sys_upper_list = [upper for [lower, upper, delta] in sys_pop_conf_lists]
        return sys_upper_list

    def estimate_pop_lower_all_systems(self, metric, alpha):
        """
        Using the provided system aggregation strategy, estimates the lower bound of the population
        of this stratum.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric.
            alpha (float): the confidence value

        Returns:
            list of float: A list with one entry per system of the lower bound of the
            population of this stratum for that system.
            Whether the estimates are floats or ints depends on the metric.

        """
        sys_pop_conf_lists = metric.estimate_population_intervals_all_systems(self, alpha)
        sys_lower_list = [lower for [lower, upper, delta] in sys_pop_conf_lists]
        return sys_lower_list

    def estimate_pop_frac_variance_all_systems(self, metric):
        """
        Estimates the population variance for each system for this stratum.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric estimating the
                population variance

        Returns:
            list of float: A list with one entry per system of the population variance of this
            stratum for that system.

        """
        return metric.estimate_pop_frac_variance_all_systems(self)

    def estimate_score_all_systems(self, metric):
        """
        Estimates the score for each system for this stratum.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric estimating the score

        Returns:
            list of float: A list with one entry per system of the score of this stratum for
            that system.

        """
        return metric.estimate_score_all_systems(self)

    def estimate_variance_all_systems(self, metric):
        """
        Estimates the variance for each system for this stratum.

        Among its uses is to compute the square of the standard error, or the score_variance
        for the strata that contains this (and other) stratum.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric estimating the variance

        Returns:
            list of float: A list with one entry per system of the variance of this stratum for
            that system.

        """
        return metric.estimate_variance_all_systems(self)

    def estimate_score_variance_all_systems(self, metric):
        """
        Estimates the score_variance for each system for this stratum.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric estimating the score variance

        Returns:
            list of float: A list with one entry per system of the score variance of this
            stratum for that system.

        """
        return metric.estimate_score_variance_all_systems(self)

    def estimate_score_variance_upper_all_systems(self, metric):
        """
        Estimates the upper bound of the score_variance for each system for this stratum; this
        upper bound is the score variance that accounts for upper and lower bound estimates
        of the population of this stratum.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric estimating the score variance

        Returns:
            list of float: A list with one entry per system of the upper score variance of
            this stratum for that system.

        """
        return metric.estimate_score_variance_upper_all_systems(self)

    def estimate_samples(self, metric, system_aggregation_strategy):
        """
        Using the provided system aggregation strategy, estimates the number of samples
        of this stratum aggregating over all systems

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric.
            system_aggregation_strategy (function, list of float -> float):
                The strategy to aggregate system values. It is a
                function that takes in a list of values and returns a single value.
                The function that aggregates floats will also aggregate integers

        Returns:
            list of int: A single aggregated number of samples across all systems

        """
        sys_samples_list = self.estimate_samples_all_systems(metric)
        if all(np.isnan(sys_samples_list)):
            return np.nan
        return system_aggregation_strategy(sys_samples_list)

    def estimate_pop(self, metric, system_aggregation_strategy):
        """
        Using the provided system aggregation strategy, estimates the population of this stratum
        aggregating over all systems

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric.
            system_aggregation_strategy (function, list of float -> float):
                The strategy to aggregate system values. It is a
                function that takes in a list of values and returns a single value.

        Returns:
            float: A single aggregated population across all systems. Because some estimates
            of population can be floats, this is sometimes a float and not an integer

        """
        sys_pop_list = self.estimate_pop_all_systems(metric)
        if all(np.isnan(sys_pop_list)):
            return np.nan
        return system_aggregation_strategy(sys_pop_list)

    def estimate_pop_upper(self, metric, system_aggregation_strategy, alpha):
        """
        Using the provided system aggregation strategy, estimates the
        upper bound of the population of this stratum
        aggregating over all systems

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric.
            system_aggregation_strategy (function, list of float -> float):
                The strategy to aggregate system values. It is a
                function that takes in a list of values and returns a single value.
            alpha (float): The (1 - confidence_level) specified

        Returns:
            float: A single aggregated upper population across all systems.

        """
        sys_pop_conf_lists = metric.estimate_population_intervals_all_systems(self, alpha)
        sys_upper_list = [upper for [lower, upper, delta] in sys_pop_conf_lists]
        if all(np.isnan(sys_upper_list)):
            return np.nan
        return system_aggregation_strategy(sys_upper_list)

    def estimate_score(self, metric, system_aggregation_strategy):
        """
        Using the provided system aggregation strategy, estimates the score of this stratum
        aggregating over all systems

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric.
            system_aggregation_strategy (function, list of float -> float):
                The strategy to aggregate system values. It is a
                function that takes in a list of values and returns a single value.

        Returns:
            float: A single aggregated score across all systems.

        """
        sys_score_list = self.estimate_score_all_systems(metric)
        if all(np.isnan(sys_score_list)):
            return np.nan
        return system_aggregation_strategy(sys_score_list)

    def estimate_variance(self, metric, system_aggregation_strategy):
        """
        Using the provided system aggregation strategy, estimates the variance of this stratum
        aggregating over all systems

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric.
            system_aggregation_strategy (function, list of float -> float):
                The strategy to aggregate system values. It is a
                function that takes in a list of values and returns a single value.

        Returns:
            float: A single aggregated variance across all systems.

        """
        sys_variance_list = self.estimate_variance_all_systems(metric)
        if all(np.isnan(sys_variance_list)):
            return np.nan
        return system_aggregation_strategy(sys_variance_list)

    def estimate_score_variance(self, metric, system_aggregation_strategy):
        """
        Using the provided system aggregation strategy, estimates the score variance of this
        stratum aggregating over all systems

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric.
            system_aggregation_strategy (function, list of float -> float):
                The strategy to aggregate system values. It is a
                function that takes in a list of values and returns a single value.

        Returns:
            float: A single aggregated score_variance across all systems.

        """
        sys_score_variance_list = self.estimate_score_variance_all_systems(metric)
        if all(np.isnan(sys_score_variance_list)):
            return np.nan
        return system_aggregation_strategy(sys_score_variance_list)

    def estimate_score_variance_upper(self, metric, system_aggregation_strategy):
        """
        Using the provided system aggregation strategy, estimates the upper score variance of this
        stratum aggregating over all systems. The upper uses upper and lower estimates of the
        population.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric.
            system_aggregation_strategy (function, list of float -> float):
                The strategy to aggregate system values. It is a
                function that takes in a list of values and returns a single value.

        Returns:
            float: A single aggregated upper score_variance across all systems.

        """
        sys_score_variance_upper_list = self.estimate_score_variance_upper_all_systems(metric)
        if all(np.isnan(sys_score_variance_upper_list)):
            return np.nan
        return system_aggregation_strategy(sys_score_variance_upper_list)
