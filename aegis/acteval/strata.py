import aegis.acteval.stratum
import math
import pandas as pd
import numpy as np
import abc
import aegis.acteval.system


class Strata(abc.ABC):
    """
    The object representing the different strata.

    Implemented as a list of Stratum objects. Within this strata contains the logic of how to
    stratify. Subclasses specify stratification strategies and strategies to aggregate
    system confidence values into one confidence value.

    Give the number of strata as -1 to indicate that we wish to have random sampling without
    stratification. This code is then stored in the self.stratify_for_pure_random class variable.
    """

    # The total number of trials, sampled or not
    num_trials = 0
    # The total number of trials useful to the metric.
    # Estimated if the metric needs to know the ground truth.
    num_trials_sampled = 0

    def __init__(self, num_strata, system_list):
        """
        Constructor to initialize strata. System objects are stored (by reference) in the strata.

        Args:
            num_strata (int): The number of strata to make
            system_list (list of :obj:`aegis.acteval.system.System`): The list of system objects.
        """
        super().__init__()
        self.num_strata = num_strata
        self.stratify_for_pure_random = False
        if num_strata == -1:
            self.num_strata = 1
            self.stratify_for_pure_random = True
        self.num_original_strata = self.num_strata
        # Initialize empty stratum for now
        self.strata = [
            aegis.acteval.stratum.Stratum(system_list) for i in range(0, self.num_strata)
        ]
        self.system_list = system_list
        self.key_df = pd.DataFrame()
        self.combined_df = pd.DataFrame()
        self.use_cached_combined_df = False
        self.score_df = pd.DataFrame()
        self.use_cached_score_df = False
        self.system_confidence_list = [[np.nan, np.nan, np.nan] for system in system_list]

    @abc.abstractmethod
    def get_strata_alpha(self, alpha):
        """
        To avoid too many type one errors, get an alpha that is adjusted accordingly for
        multiple systems

        Args:
            alpha: (float) The original desired alpha

        Returns:
            float: new_alpha, The new desired alpha

        """
        pass

    def dirty_strata_cache(self):
        """
        Mark all cached variables as dirty for future computations. Calls the stratum
        :meth:aegis.acteval.stratum.Stratum.dirty_strata_cache method to dirty the caches
        of each stratum.

        Returns:
            Nothing

        """
        self.use_cached_combined_df = False
        self.use_cached_score_df = False
        # Dirty cache of each stratum
        [stratum.dirty_stratum_cache() for stratum in self.strata]
        return

    def get_combined_systems_df(self):
        """
        Takes the system data frames with system scores and combines to get one data frame
        that also includes the stratum indices

        If the "stratum_index" field has already been added to the system data frames, then
        that field will be in the returned data frame. Otherwise, if the field "stratum_index"
        has not yet been added, then the "stratum_index" field will not appear in this combined
        data frame.

        Returns:
            pandas.core.frame.DataFrame: output_df a copy of the combined data frame,
            where each system_id is a column whose value is the score.

        """
        if self.use_cached_combined_df:
            return self.combined_df

        # stratum_combined_df_list = [stratum.get_combined_systems_df()
        #                             for stratum in self.strata]
        #
        # combined_df = stratum_combined_df_list[0].copy(deep=True)
        # # If we have yet to stratify, each stratum is the entire df, and we want no duplicates
        # if 'stratum_index' in combined_df:
        #     for i in range(1, len(stratum_combined_df_list)):
        #         combined_df = combined_df.append(stratum_combined_df_list[i], sort=True)
        # self.combined_df = combined_df
        # self.use_cached_combined_df = True
        # return combined_df

        combined_df = self.strata[0].get_combined_systems_df().copy(deep=True)
        if 'stratum_index' in combined_df:
            for stratum in self.strata[1:]:
                stratum_comb_df = stratum.get_combined_systems_df()
                combined_df = combined_df.append(stratum_comb_df, sort=True)
        self.combined_df = combined_df
        self.use_cached_combined_df = True
        return combined_df

    def get_combined_systems_score_df(self):
        """
        Takes the system data frames with system scores and combines to get one data frame
        that also includes the stratum indices. Only returns subset where we have the key.

        If the "stratum_index" field has already been added to the system data frames, then
        that field will be in the returned data frame. Otherwise, if the field "stratum_index"
        has not yet been added, then the "stratum_index" field will not appear in this combined
        data frame.

        Returns:
            pandas.core.frame.DataFrame: output_df a copy of the combined data frame, where each
            system_id is a column
            whose value is the score.

        """
        if self.use_cached_score_df:
            return self.score_df

        stratum_combined_df_list = [stratum.get_combined_systems_score_df()
                                    for stratum in self.strata]

        combined_score_df = stratum_combined_df_list[0].copy(deep=True)
        if 'stratum_index' in combined_score_df:
            for i in range(1, len(stratum_combined_df_list)):
                combined_score_df = combined_score_df.append(stratum_combined_df_list[i], sort=True)
        self.score_df = combined_score_df
        self.use_cached_score_df = True
        return combined_score_df

    @abc.abstractmethod
    def stratify(self):
        """
        Produce strata according to a stratification strategy, which is defined by the
        implementing class.

        Returns:
            aegis.acteval.strata.Strata: new_strata, the strata object after stratification

        """
        pass

    def find_needed_initial_samples(self, metric, initial_samples, rng):
        """
        In order to cover the case where we don't sample from each stratum necessary edge cases
        (Such as when the metric is a combination of two components) and to ensure samples
        from each stratum, this method will find such examples and will sample at random.

        This will work if initial samples have already been added. This will then find the gaps.
        This requires at least two samples per strata, and more depending on the metric

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric to score systems
            initial_samples (int): The number of initial samples per bin to request
            rng (:obj:`numpy.random.RandomState`): random state object with stored random state

        Returns:
            list of object: A list of trial_id values to sample

        """
        samples_list = []
        initial_samples_per_bin = [round(initial_samples / len(self.strata))] * len(self.strata)
        diff_samp = initial_samples - sum(initial_samples_per_bin)
        for i in range(0, abs(diff_samp)):
            initial_samples_per_bin[i] = initial_samples_per_bin[i] + np.sign(diff_samp)
        j = 0
        for stratum in self.strata:
            samples_list.extend(stratum.find_needed_initial_samples(metric,
                                                                    initial_samples_per_bin[j],
                                                                    rng))
            j += 1

        return list(set(samples_list))

    def add_samples_to_strata(self, samples_df):
        """
        Takes in a set of scored samples and updates the strata to now have the keys for those
        samples.

        Args:
            samples_df (obj:`pandas.core.frame.DataFrame`): a dataframe with two columns
                (trial_id, key), and a key value for all
                of the trial_id values provided. This frame will not have all trial ids and
                will only have the trial_id values of the trials that are scored

        Returns:
            aegis.acteval.strata.Strata: The updated strata object
        """
        if samples_df is None:
            return
        if samples_df.shape[0] == 0:
            return

        self.key_df = self.key_df.merge(
            samples_df, how="left", on=["trial_id"]
        )
        self.key_df.loc[
            pd.isna(self.key_df["key_x"]), "key_x"
        ] = self.key_df.loc[pd.isna(self.key_df["key_x"]), "key_y"]
        self.key_df.drop(["key_y"], axis=1, inplace=True)
        self.key_df.rename(columns={"key_x": "key"}, inplace=True)
        # Merge data frame with each strata
        self.strata = [
            stratum.add_samples_to_stratum(samples_df) for stratum in self.strata
        ]
        # TODO: Check that we have not yet added previously-taken samples.

        # Add the number of trials to those sampled
        self.num_trials_sampled += samples_df.shape[0]
        # Mark cached values as dirty when we add samples
        self.dirty_strata_cache()
        return self

    def estimate_samples_all_systems(self, metric):
        """
        Returns the total number of tried samples to include according to the metric. Updates
        System objects with the result for each system.

        Does so by incorporating the sums of the populations for each stratum, which then
        send their sets of trials to the metric provided.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines
                how trials are scored

        Returns:
            list of int: A list of number of sampled trials, one per system

        """

        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric)
            for stratum in self.strata
        ]
        samples_list = [0 for sys in range(0, len(self.system_list))]
        for ind in range(0, len(self.system_list)):
            curr_strata_samples = [
                stratum_samples_list[ind] for stratum_samples_list in strata_samples_list
            ]
            sys_samples = np.nansum(curr_strata_samples)
            samples_list[ind] = sys_samples
        # Now store the number of samples with the system
        for system, samples in zip(self.system_list, samples_list):
            system.sampled_trials = samples
        return samples_list

    def estimate_pop_all_systems(self, metric):
        """
        Returns the total number of samples that are included in the population for the metric.
        Updates System objects with the result for each system.

        Does so by incorporating the sums of the populations for each stratum, which then
        send their sets of trials to the metric provided

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines
                how trials are scored

        Returns:
            list of float: A list of population sizes, one list per system. The results
            may be integers or floats depending on the metric used.

        """

        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric)
            for stratum in self.strata
        ]
        pop_list = [0 for sys in range(0, len(self.system_list))]
        for ind in range(0, len(self.system_list)):
            curr_strata_pop = [
                stratum_pop_list[ind] for stratum_pop_list in strata_pop_list
            ]
            sys_pop = np.nansum(curr_strata_pop)
            pop_list[ind] = sys_pop
        # Now store the populations with the systems
        for system, pop in zip(self.system_list, pop_list):
            system.population = pop
        return pop_list

    def estimate_pop_frac_all_systems(self, metric):
        """
        Returns the fraction of stratum trials relevant for the population.
        Calls estimate_pop_all_systems

        Does so by incorporating the sums of the populations for each stratum, which then
        send their sets of trials to the metric provided

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines how
                trials are scored

        Returns:
            list of float: A list of population fraction sizes, one list per system

        """

        pop_frac_list = self.estimate_pop_all_systems(metric)
        pop_frac_list = [pf / self.num_trials for pf in pop_frac_list]
        # Now store the populations with the systems
        for system, pop in zip(self.system_list, pop_frac_list):
            system.population_frac = pop
        return pop_frac_list

    def estimate_pop_frac_variance_all_systems(self, metric):
        """
        Returns the variance of the fraction population relevant estimate for the metric.
        Updates System objects with the result for each system.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines how
                trials are scored

        Returns:
            list of float: A list of population fraction variance sizes, one list per system

        """

        # Very important: check metric population size; need to adjust for different metrics
        strata_pop_frac_var_list = [
            stratum.estimate_pop_frac_variance_all_systems(metric)
            for stratum in self.strata
        ]
        # Samples used for finite population correction
        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric)
            for stratum in self.strata
        ]
        strata_trial_size_list = [stratum.num_trials for stratum in self.strata]
        pop_var_list = [0 for sys in range(0, len(self.system_list))]
        for ind in range(0, len(self.system_list)):
            curr_strata_pop_frac_var = [
                stratum_pop_frac_var_list[ind]
                for stratum_pop_frac_var_list in strata_pop_frac_var_list
            ]
            curr_strata_samples = [
                stratum_samples_list[ind]
                for stratum_samples_list in strata_samples_list
            ]
            sys_pop_var = 0
            try:
                # Right now weigh each estimate of the population fraction
                # equally
                for (stratum_pop_frac_var, stratum_samples, strata_size) in \
                        zip(curr_strata_pop_frac_var, curr_strata_samples, strata_trial_size_list):
                    if stratum_pop_frac_var > 0:
                        curr_pop_var = (((strata_size/self.num_trials) ** 2) *
                                        stratum_pop_frac_var)
                        # multiply by fpc
                        if stratum_samples >= 1 and strata_size > 1:
                            curr_pop_var = curr_pop_var * \
                                         (1 - (stratum_samples - 1)/(strata_size - 1))
                    else:
                        #  stratum_pop_frac_var == 0
                        curr_pop_var = 0
                    sys_pop_var += curr_pop_var
            except ZeroDivisionError:
                sys_pop_var = 0
            pop_var_list[ind] = sys_pop_var
        # Now store the population variances with the systems
        for system, pop_var in zip(self.system_list, pop_var_list):
            system.population_frac_variance = pop_var
        return pop_var_list

    def estimate_pop_variance_all_systems(self, metric):
        """
        Returns the population variance for each system based on the metric.
        Updates System objects with the result for each system.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines how
                trials are scored

        Returns:
            list of float: A list of population variance sizes, one list per system

        """
        pop_var_list = self.estimate_pop_frac_variance_all_systems(metric)
        pop_var_list = [pv * self.num_trials * self.num_trials for pv in pop_var_list]
        # Now store the populations with the systems
        for system, pop in zip(self.system_list, pop_var_list):
            system.population_variance = pop
        return pop_var_list

    def estimate_score_all_systems(self, metric):
        r"""
        Returns the score of the entire strata according to the metric.
        Updates System objects with the result for each system.

        This leverages the computation of the score for each stratum according to the metric,
        and then uses standard stratified sampling to estimate the score, or the `population mean`.
        The formula used is:

        .. math::
            score\_hat =
            \frac{\sum_{l=1}^{self.num\_strata}(N_{L}*stratum.estimate\_metric\_score)}{N}

        where there are :math:`L` strata, and :math:`N_L` is the total number of trials in that
        strata and :math:`N` is the total number of samples in the population.


        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines how
                trials are scored

        Returns:
            list of float: A list of scores, one score per system.

        """

        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric)
            for stratum in self.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(metric)
            for stratum in self.strata
        ]
        score_list = [0 for sys in range(0, len(self.system_list))]
        for ind in range(0, len(self.system_list)):
            curr_strata_pop = [
                stratum_pop_list[ind] for stratum_pop_list in strata_pop_list
            ]
            curr_strata_score = [
                stratum_score_list[ind] for stratum_score_list in strata_score_list
            ]
            ind_strata_pop = self.estimate_pop_all_systems(metric)[
                ind
            ]
            try:
                # We need to know when we don't know the score, so we don't remove NAs
                sys_score = 0
                for (stratum_pop, stratum_score) in zip(curr_strata_pop, curr_strata_score):
                    if stratum_pop >= 1:
                        stratum_score = (stratum_pop / ind_strata_pop) * stratum_score
                    elif stratum_pop == 0:
                        stratum_score = 0
                    sys_score = np.sum([sys_score, stratum_score])
                    if np.isnan(stratum_score):
                        sys_score = np.nan
                        break
            except ZeroDivisionError:
                sys_score = 0
            score_list[ind] = sys_score
        # Now store the scores with the systems
        for system, score in zip(self.system_list, score_list):
            system.score = score
        return score_list

    def estimate_score_upper_all_systems(self, metric, alpha):
        r"""
        Returns the score of the entire strata according to the metric.
        Updates System objects with the result for each system. This score estimate gives
        the upper score based on combining the strata using the lower bound or the upper bound
        of the population estimate depending on which gives the higher score

        This leverages the computation of the score for each stratum according to the metric,
        and then uses standard stratified sampling to estimate the score, or the `population mean`.
        The formula used is:

        .. math::
            score\_hat =
            \frac{\sum_{l=1}^{self.num\_strata}(N_{L}*stratum.estimate\_metric\_score)}{N}

        where there are :math:`L` strata, and :math:`N_L` is the total number of trials in that
        strata and :math:`N` is the total number of samples in the population.


        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines how
                trials are scored
            alpha (float): The specified alpha for confidence

        Returns:
            list of float: A list of score values based on different population upper and
            lower bounds, one per system

        """

        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric)
            for stratum in self.strata
        ]
        strata_pop_upper_list = [
            stratum.estimate_pop_upper_all_systems(metric, alpha)
            for stratum in self.strata
        ]
        strata_pop_lower_list = [
            stratum.estimate_pop_lower_all_systems(metric, alpha)
            for stratum in self.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(metric)
            for stratum in self.strata
        ]
        score_list = [0 for sys in range(0, len(self.system_list))]
        for ind in range(0, len(self.system_list)):
            curr_strata_pop = [
                stratum_pop_list[ind] for stratum_pop_list in strata_pop_list
            ]
            curr_strata_pop_upper = [
                stratum_pop_upper_list[ind] for stratum_pop_upper_list in strata_pop_upper_list
            ]
            curr_strata_pop_lower = [
                stratum_pop_lower_list[ind] for stratum_pop_lower_list in strata_pop_lower_list
            ]
            curr_strata_score = [
                stratum_score_list[ind] for stratum_score_list in strata_score_list
            ]
            ind_strata_pop = self.estimate_pop_all_systems(metric)[
                ind
            ]
            # Use previously cached system average score to determine
            average_score = self.system_list[ind].score
            # Change curr_strata_pop_list depending on the score
            for sind in range(0, len(self.strata)):
                if curr_strata_pop[sind] == 0:
                    pass
                elif curr_strata_score[sind] >= average_score:
                    curr_strata_pop[sind] = min(curr_strata_pop_upper[sind],
                                                self.strata[sind].num_trials)
                else:
                    curr_strata_pop[sind] = max(curr_strata_pop_lower[sind], 0)
            try:
                # We need to know when we don't know the score, so we don't remove NAs
                sys_score = 0
                for (stratum_pop, stratum_score) in zip(curr_strata_pop, curr_strata_score):
                    if stratum_pop >= 1:
                        stratum_score = (stratum_pop / ind_strata_pop) * stratum_score
                    elif stratum_pop == 0:
                        stratum_score = 0
                    sys_score = np.sum([sys_score, stratum_score])
                    if np.isnan(stratum_score):
                        sys_score = np.nan
                        break
            except ZeroDivisionError:
                sys_score = 0
            score_list[ind] = sys_score
        # Now store the scores with the systems
        for system, score in zip(self.system_list, score_list):
            system.score_upper = score
        return score_list

    def estimate_score_lower_all_systems(self, metric, alpha):
        r"""
        Returns the score of the entire strata according to the metric.
        Updates System objects with the result for each system. This score estimate gives
        the lower score based on combining the strata using the lower bound or the upper bound
        of the population estimate depending on which gives the lower score

        This leverages the computation of the score for each stratum according to the metric,
        and then uses standard stratified sampling to estimate the score, or the `population mean`.
        The formula used is:

        .. math::
            score\_hat =
            \frac{\sum_{l=1}^{self.num\_strata}(N_{L}*stratum.estimate\_metric\_score)}{N}

        where there are :math:`L` strata, and :math:`N_L` is the total number of trials in that
        strata and :math:`N` is the total number of samples in the population.


        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines how
                trials are scored
            alpha (float): The specified alpha for confidence

        Returns:
            list of float: A list of score values based on different population upper and
            lower bounds, one per system

        """

        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric)
            for stratum in self.strata
        ]
        strata_pop_upper_list = [
            stratum.estimate_pop_upper_all_systems(metric, alpha)
            for stratum in self.strata
        ]
        strata_pop_lower_list = [
            stratum.estimate_pop_lower_all_systems(metric, alpha)
            for stratum in self.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(metric)
            for stratum in self.strata
        ]
        score_list = [0 for sys in range(0, len(self.system_list))]
        for ind in range(0, len(self.system_list)):
            curr_strata_pop = [
                stratum_pop_list[ind] for stratum_pop_list in strata_pop_list
            ]
            curr_strata_pop_upper = [
                stratum_pop_upper_list[ind] for stratum_pop_upper_list in strata_pop_upper_list
            ]
            curr_strata_pop_lower = [
                stratum_pop_lower_list[ind] for stratum_pop_lower_list in strata_pop_lower_list
            ]
            curr_strata_score = [
                stratum_score_list[ind] for stratum_score_list in strata_score_list
            ]
            ind_strata_pop = self.estimate_pop_all_systems(metric)[
                ind
            ]
            # Use previously cached system average score to determine
            average_score = self.system_list[ind].score
            # Change curr_strata_pop_list depending on the score
            for sind in range(0, len(self.strata)):
                if curr_strata_pop[sind] == 0:
                    pass
                elif curr_strata_score[sind] >= average_score:
                    curr_strata_pop[sind] = max(curr_strata_pop_lower[sind], 0)
                else:
                    curr_strata_pop[sind] = min(curr_strata_pop_upper[sind],
                                                self.strata[sind].num_trials)
            try:
                # We need to know when we don't know the score, so we don't remove NAs
                sys_score = 0
                for (stratum_pop, stratum_score) in zip(curr_strata_pop, curr_strata_score):
                    if stratum_pop >= 1:
                        stratum_score = (stratum_pop / ind_strata_pop) * stratum_score
                    elif stratum_pop == 0:
                        stratum_score = 0
                    sys_score = np.sum([sys_score, stratum_score])
                    if np.isnan(stratum_score):
                        sys_score = np.nan
                        break
            except ZeroDivisionError:
                sys_score = 0
            score_list[ind] = sys_score
        # Now store the scores with the systems
        for system, score in zip(self.system_list, score_list):
            system.score_lower = score
        return score_list

    def estimate_score_variance_all_systems(self, metric):
        r"""
        Uses stratified sampling to estimate the metric variance.
        Updates System objects with the result for each system.

        The formula to take the different stratum variances and provide a combined variance is
        taken from pg 229–230 of "Mathematical Statistics and Data Analysis" by John Rice,
        3rd Edition. The formula is implemetned without the finite population correction.
        This formula is:

        .. math::
            \widehat{score} =
            \sum_{i=1}^{self.num\_strata}\left(W_{l}^2*
            score\_var_l*(1 - \frac{n_l - 1}{N_l - 1})\right)

        with

        .. math::
            fpc = 1 - \frac{n_l - 1}{N_l - 1}

        as a finite population correction.

        Where for stratum :math:`l`, :math:`score\_var_l` is the score variance of that stratum.
        Rather than
        adding the finite population correction within the stratum, we add the finite population
        correction here. The finite population correction is essential for telling the sampler
        that exhausted strata mean that our measurement error has decreased because we have all
        of the samples and cannot sample further. Else we would be in an infinite loop trying
        to sample exhausted stratum. The name
        components score_variance and variance are used to distinguish when the method is
        computing the variance or the score variance (the square of the standard error of the
        score). We use the score variance so that advance score_variance and confidence interval
        corrections can be specified by the metric.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines how
                trials are scored

        Returns:
            list of float: A list of variances of the score measurements,
            one score_variance per system

        """

        # Very important: check metric population size; need to adjust for different metrics
        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric)
            for stratum in self.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric)
            for stratum in self.strata
        ]
        strata_score_var_list = [
            stratum.estimate_score_variance_all_systems(metric)
            for stratum in self.strata
        ]
        score_var_list = [0 for sys in range(0, len(self.system_list))]
        for ind in range(0, len(self.system_list)):
            curr_strata_samples = [
                stratum_samples_list[ind] for stratum_samples_list in strata_samples_list
            ]
            curr_strata_pop = [
                stratum_pop_list[ind] for stratum_pop_list in strata_pop_list
            ]
            curr_strata_score_var = [
                stratum_score_var_list[ind] for stratum_score_var_list in strata_score_var_list
            ]
            ind_strata_pop = self.estimate_pop_all_systems(metric)[ind]
            try:
                # Using finite population correction in strata as
                # (1 - (stratum_samples - 1)/(stratum_pop - 1))
                # We need the missing variances, so keep the na values
                sys_score_var = 0
                for (stratum_pop, stratum_score_var, stratum_samples) in zip(
                        curr_strata_pop, curr_strata_score_var, curr_strata_samples):
                    if stratum_pop > 1:
                        curr_score_var = (((stratum_pop / ind_strata_pop) ** 2) *
                                          stratum_score_var) * \
                                         (1 - (stratum_samples - 1)/(stratum_pop - 1))
                    elif stratum_pop == 0:
                        curr_score_var = 0
                    else:
                        curr_score_var = (((stratum_pop / ind_strata_pop) ** 2) * stratum_score_var)
                    sys_score_var += curr_score_var
            except ZeroDivisionError:
                sys_score_var = 0
            score_var_list[ind] = sys_score_var
        # Now store the variances with the systems
        for system, score_var in zip(self.system_list, score_var_list):
            system.score_variance = score_var
        return score_var_list

    def estimate_score_variance_upper_all_systems(self, metric, alpha):
        r"""
        Uses stratified sampling to estimate the metric variance assuming upper bounds on
        population estimates. This also gives a nonzero uncertainty when the population is 0
        but the upper estimate of the population is above 0.
        Updates System objects with the result for each system.

        This score variance estimate gives
        the upper score variance based on combining the strata using the lower bound or the
        upper bound of the population estimate depending on which gives the higher variance

        The formula to take the different stratum variances and provide a combined variance is
        taken from pg 229–230 of "Mathematical Statistics and Data Analysis" by John Rice,
        3rd Edition. The formula is implemetned without the finite population correction.
        This formula is:

        .. math::
            \widehat{score} =
            \sum_{i=1}^{self.num\_strata}\left(W_{l}^2*
            score\_var_l*(1 - \frac{n_l - 1}{N_l - 1})\right)

        with

        .. math::
            fpc = 1 - \frac{n_l - 1}{N_l - 1}

        as a finite population correction.

        Where for stratum :math:`l`, :math:`score\_var_l` is the score variance of that stratum.
        Rather than
        adding the finite population correction within the stratum, we add the finite population
        correction here. The finite population correction is essential for telling the sampler
        that exhausted strata mean that our measurement error has decreased because we have all
        of the samples and cannot sample further. Else we would be in an infinite loop trying
        to sample exhausted stratum. The name
        components score_variance and variance are used to distinguish when the method is
        computing the variance or the score variance (the square of the standard error of the
        score). We use the score variance so that advance score_variance and confidence interval
        corrections can be specified by the metric.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object that determines how
                trials are scored
            alpha (float): The confidence bound for upper population estimation

        Returns:
            list of float: A list of variances of the score measurements,
            one score_variance per system

        """

        # Very important: check metric population size; need to adjust for different metrics
        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric)
            for stratum in self.strata
        ]
        strata_pop_upper_list = [
            stratum.estimate_pop_upper_all_systems(metric, alpha)
            for stratum in self.strata
        ]
        strata_score_var_upper_list = [
            stratum.estimate_score_variance_upper_all_systems(metric)
            for stratum in self.strata
        ]
        score_var_upper_list = [0 for sys in range(0, len(self.system_list))]
        for ind in range(0, len(self.system_list)):
            curr_strata_samples = [
                stratum_samples_list[ind] for stratum_samples_list in strata_samples_list
            ]
            curr_strata_pop_upper = [
                stratum_pop_upper_list[ind] for stratum_pop_upper_list in strata_pop_upper_list
            ]
            curr_strata_score_var_upper = [
                stratum_score_var_upper_list[ind]
                for stratum_score_var_upper_list in strata_score_var_upper_list
            ]
            ind_strata_pop = self.estimate_pop_all_systems(metric)[ind]
            try:
                # Using finite population correction in strata as
                # (1 - (stratum_samples - 1)/(stratum_pop - 1))
                # We need the missing variances, so keep the na values
                sys_score_var_upper = 0
                for (stratum_pop_upper, stratum_score_var_upper, stratum_samples) in zip(
                        curr_strata_pop_upper, curr_strata_score_var_upper, curr_strata_samples):
                    if stratum_pop_upper > 1:
                        curr_score_var_upper = (((stratum_pop_upper / ind_strata_pop) ** 2) *
                                                stratum_score_var_upper) * \
                                                (1 - (stratum_samples - 1)/(stratum_pop_upper - 1))
                    elif stratum_pop_upper == 0:
                        curr_score_var_upper = 0
                    else:
                        curr_score_var_upper = (((stratum_pop_upper / ind_strata_pop) ** 2) *
                                                stratum_score_var_upper)
                    sys_score_var_upper += curr_score_var_upper
            except ZeroDivisionError:
                sys_score_var_upper = 0
            score_var_upper_list[ind] = sys_score_var_upper
        # Now store the variances with the systems
        for system, score_var_upper in zip(self.system_list, score_var_upper_list):
            system.score_variance_upper = score_var_upper
        return score_var_upper_list

    def get_confidence_intervals_all_systems(self, metric, alpha):
        r"""
        Gets a confidence interval for each system in this strata, represented as a list of lists
        per system.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object
            alpha (float): the desired probability level to get (1-alpha) confidence for

        Returns:
            list of list of float: A list of lists, where each entry of the list is a
            three-element list,
            [lower_value, upper_value, delta], where the delta is
            usually the higher of (score - lower_value) and (higher_value - score).

        """
        conf_list = metric.get_confidence_intervals_all_systems(self, alpha)
        self.system_confidence_list = conf_list
        return conf_list

    def get_confidence_intervals_true_pop_all_systems(self, metric, alpha):
        r"""
        Gets a confidence interval for each system in this strata, represented as a list of lists
        per system. This estimate assumes no uncertainty in the population estimates.

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): the metric object
            alpha (float): the desired probability level to get (1-alpha) confidence for

        Returns:
            list of list of float: A list of lists, where each entry of the list is a
            three-element list,
            [lower_value, upper_value, delta], where the delta is
            usually the higher of (score - lower_value) and (higher_value - score).

        """
        conf_list = metric.get_confidence_intervals_true_pop_all_systems(self, alpha)
        return conf_list

    @abc.abstractmethod
    def aggregate_system_confidence_values(self):
        """
        Produces an aggregated system confidence value according to the strategy of the Strata
        subclass.

        Returns:
            float: An aggregated confidence value

        """
        pass

    @abc.abstractmethod
    def aggregate_system_stats(self, sys_stat_list):
        """
        Given a list of statistics with one per system, aggregates them to produce a single
        value.

        Args:
            sys_stat_list (list of float): The list of values, one value per system, to aggregate.

        Returns:
            float: An aggregated value.

        """
        pass

    def estimate_pop(self, metric):
        """
        Using the aggregation strategy defined by the strata class, aggregates the system's
        populations to determine the population of the strata

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric that determines the
                value for each system

        Returns:
            A float: single population value. Can be an integer or float depending on the
            metric object.

        """
        sys_pop_list = self.estimate_pop_all_systems(metric)
        return self.aggregate_system_stats(sys_pop_list)

    def estimate_score(self, metric):
        """
        Using the aggregation strategy defined by the strata class, aggregates the system's
        populations to determine the score of the strata

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric that determines the value
                for each system

        Returns:
            float: A single score value.

        """
        sys_score_list = self.estimate_score_all_systems(metric)
        return self.aggregate_system_stats(sys_score_list)

    def estimate_score_variance(self, metric):
        """
        Using the aggregation strategy defined by the strata class, aggregates the system's
        populations to determine the score_variance (square of the standard error) of the strata

        Args:
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric that determines the value
                for each system

        Returns:
            float: A single score_variance value.

        """
        sys_variance_list = self.estimate_score_variance_all_systems(metric)
        return self.aggregate_system_stats(sys_variance_list)


class StrataFirstSystem(Strata):
    """
    Default Stratification Style that stratifies according to the first system. It is recommended
    to specify the system ordering in aegis.data_processor.extract_files_from_directory() method
    so that the desired system is used.

    This class is useful for single-system stratification as well as tests that stratify by one
    system only.

    The stratification stratifies only by the first system, and the system aggregator always
    takes the statistic of the first system. When sampling, we can customize whether the bin
    widths of the strata should be equal, or if the bin widths should be chosen so that an equal
    number of trials is in each bin (Each stratum has the same size).
    """

    def __init__(self, num_strata, system_list):
        """
        Constructor to initialize strata. System objects are stored (by reference) in the strata.
        Calls superclass constructor and does nothing else.

        Args:
            num_strata (int): The number of strata to make
            system_list (list of :obj:`aegis.acteval.system.System`): The list of system objects.
        """
        super().__init__(num_strata, system_list)
        self.num_strata = num_strata
        self.stratify_for_pure_random = False
        if num_strata == -1:
            self.num_strata = 1
            self.stratify_for_pure_random = True

    def get_strata_alpha(self, alpha):
        """
        To avoid too many type one errors, get an alpha that is adjusted accordingly for
        multiple systems. This class ignores all but one system, so return just one system

        Args:
            alpha: (float) The original desired alpha

        Returns:
            float: new_alpha, The new desired alpha

        """
        new_alpha = alpha
        return new_alpha

    def stratify(self, bin_style='equal'):
        """
        Stratification method that stratifies purely by the first system and ignores
        other systems.

        Since this ignores the trial data, there is no trial_df argument.

        Start with the easiest strategy: stratify by score such that the bin widths (the score
        ranges, not the number of items in each bin, are the same), and stratify by the score of
        the first system.

        In this strata of stratum, the population size
        is the total number of trials in all of the stratum, not the total number of trials that
        have been sampled. For metrics such as recall where this cannot be computed directly,
        this will be estimated by by incorporating the estimates of each stratum in the
        stratified sampling manner.

        Args:
            bin_style (str): The style to stratify the bins. Default is 'equal'.
                'equal': Stratify the bins so that the range of values is equal, or the bins
                are of equal width
                'perc': Stratfiy by percentile, or so that an equal number of trials are in each bin

        Returns:
            aegis.acteval.strata.Strata: new_strata, the strata object after stratification

        """
        # We obtain the combined data frame from the systems
        combined_df = self.get_combined_systems_df()
        self.dirty_strata_cache()
        self.num_trials = combined_df.shape[0]

        # column 0 is 'trial_id', 1 is the first system
        first_sys_id = self.system_list[0].system_id
        # For now just stratify by the first system

        if self.stratify_for_pure_random:
            combined_df["stratum_index"] = 0
        else:
            # Cut by percentage or by width
            num_values = len(combined_df[str(first_sys_id)].unique())
            if bin_style == 'perc':
                combined_df["stratum_bin"] = pd.qcut(
                    combined_df[first_sys_id], np.nanmin([num_values, self.num_strata]),
                    duplicates='drop'
                )
            else:
                # Default:  cut bins into equal widths
                combined_df["stratum_bin"] = pd.cut(
                    combined_df[first_sys_id], np.nanmin([num_values, self.num_strata]),
                    duplicates='drop'
                )

            # This counts on the stratum_index mapping to be from 0 to (num_strata - 1)
            combined_df["stratum_index"] = combined_df["stratum_bin"].cat.codes
            combined_df.drop(columns=["stratum_bin"], inplace=True)

        combined_df["key"] = np.nan

        # Check if we need to construct fewer strata then specified
        # Rebuild stratum objects with fewer strata objects
        num_nonempty_strata = len(combined_df['stratum_index'].value_counts().index.to_list())
        # This covers any stratum differences
        if num_nonempty_strata != self.num_strata:
            self.num_strata = num_nonempty_strata
            self.strata = [
                aegis.acteval.stratum.Stratum(self.system_list) for i in range(0, self.num_strata)
            ]

        # We need the index number to filter the stratum, so we use a for loop
        non_empty_stratum_index_list = combined_df['stratum_index'].value_counts().index.to_list()
        non_empty_stratum_index_list.sort()
        for i in range(0, self.num_strata):
            curr_stratum = self.strata[i]
            stratum_ind = non_empty_stratum_index_list[i]
            stratum_key_df = combined_df.loc[combined_df["stratum_index"] == stratum_ind, :]
            # Stratum method makes a deep copy of this method
            curr_stratum.construct_stratum_from_trials(
                stratum_key_df.loc[:, ["trial_id", "stratum_index", "key"]], stratum_ind)

        # Now store the key df with the stratum index, and push that information to the system
        self.key_df = combined_df.loc[:, ["trial_id", "stratum_index", "key"]]

        [system.add_stratum_index_to_system_data(combined_df.loc[:, ["trial_id", "stratum_index"]])
         for system in self.system_list]
        self.dirty_strata_cache()
        return self

    def aggregate_system_confidence_values(self):
        """
        Produces an aggregated system confidence value by taking the confidence value of the first
        system and ignoring all other systems' values.

        Returns:
            float: An aggregated confidence value

        """
        delta = self.system_confidence_list[0][2]
        return delta

    def aggregate_system_stats(self, sys_stat_list):
        """
        Aggregates the system stats list by taking the first system's value. Ignores values
        of all other systems.

        Args:
            sys_stat_list (list of float): The list of values, one value per system, to aggregate.

        Returns:
            float: An aggregated value.

        """
        return sys_stat_list[0]


class StrataMultiSystemIntersect(Strata):
    r"""
    Default Stratification Style that stratifies according to all of the systems. It creates
    :math:`\sqrt[num\_systems]{num\_strata}` per system to get as close as possible to
    handle floating point errors

    The system aggregates such that all systems have to be within their confidence for
    a successful round. Other stats are averaged.
    """

    def __init__(self, num_strata, system_list):
        r"""
        Constructor to initialize strata. System objects are stored (by reference) in the strata.
        Calls superclass constructor and does nothing else. It creates
        :math:`\sqrt[num\_systems]{num\_strata}` per system

        Args:
            num_strata (int): The number of strata to make
            system_list (list of :obj:`aegis.acteval.system.System`): The list of system objects.
        """
        self.num_strata = num_strata
        self.stratify_for_pure_random = False
        if num_strata == -1:
            self.num_strata = 1
            self.stratify_for_pure_random = True
        self.num_systems = len(system_list)
        num_strata_per_system = math.pow(self.num_strata, 1.0/len(system_list))
        self.num_strata_per_system = int(np.round(num_strata_per_system))
        # Correct the number of strata in case of a mistake or roundoff
        corr_num_strata = int(np.round(self.num_strata_per_system ** (len(system_list))))
        super().__init__(corr_num_strata, system_list)
        self.stratify_for_pure_random = False
        if num_strata == -1:
            self.num_strata = 1
            self.stratify_for_pure_random = True

    def get_strata_alpha(self, alpha):
        """
        To avoid too many type one errors, get an alpha that is adjusted accordingly for
        multiple systems. This class uses all systems, so find an alpha whose union bound
        gives us the desired alpha.

       Args:
            alpha: (float) The original desired alpha

        Returns:
            float: new_alpha, The new desired alpha

        """
        new_alpha = 1 - np.power(1 - alpha, 1/self.num_systems)
        return new_alpha

    def stratify(self, bin_style='equal'):
        r"""
        Stratification method that stratifies purely by all of the systems. It creates
        :math:`\sqrt[num\_systems]{num\_strata}` strata for each system that are binned according
        to the specified bin style. Then it intersects the strata across each system to produce
        disjoint strata for sampling.

        Since this ignores the trial data, there is no trial_df argument.

        Args:
            bin_style (str): The style to stratify the bins. Default is 'equal'.
                'equal': Stratify the bins so that the range of values is equal, or the bins
                are of equal width
                'perc': Stratfiy by percentile, or so that an equal number of trials are in each bin

        Returns:
            aegis.acteval.strata.Strata: new_strata, the strata object after stratification

        """
        # We obtain the combined data frame from the systems
        combined_df = self.get_combined_systems_df()
        self.dirty_strata_cache()
        self.num_trials = combined_df.shape[0]

        # preserve ordering provided to us
        sys_id_list = [system.system_id for system in self.system_list]

        # First make a stratum_index for each system

        combined_df["stratum_index"] = 0
        if not self.stratify_for_pure_random:
            for sys_id in sys_id_list:
                sys_ind = sys_id_list.index(sys_id)
                # Cut by percentage or by width
                num_sys_values = len(combined_df[str(sys_id)].unique())
                if bin_style == 'perc':
                    combined_df[str(sys_id) + "_stratum_bin"] = pd.qcut(
                        combined_df[sys_id], np.nanmin([num_sys_values,
                                                        self.num_strata_per_system]),
                        duplicates='drop'
                    )
                else:
                    # Default:  cut bins into equal widths
                    combined_df[str(sys_id) + "_stratum_bin"] = pd.cut(
                        combined_df[sys_id], np.nanmin([num_sys_values,
                                                        self.num_strata_per_system]),
                        duplicates='drop'
                    )
                # This counts on the stratum_index mapping to be from 0 to (num_strata - 1)
                combined_df[str(sys_id) + "_stratum_index"] = \
                    combined_df[str(sys_id) + "_stratum_bin"].cat.codes
                combined_df["stratum_index"] += (self.num_strata_per_system**sys_ind) * \
                                                (combined_df[str(sys_id) + "_stratum_index"])
                combined_df.drop(columns=[str(sys_id) + "_stratum_bin"], inplace=True)
                combined_df.drop(columns=[str(sys_id) + "_stratum_index"], inplace=True)

        combined_df["key"] = np.nan

        # Check if we need to construct fewer strata then specified
        # Rebuild stratum objects with fewer strata objects
        num_nonempty_strata = len(combined_df['stratum_index'].value_counts().index.to_list())
        # This covers any stratum differences
        if num_nonempty_strata != self.num_strata:
            self.num_strata = num_nonempty_strata
            self.strata = [
                aegis.acteval.stratum.Stratum(self.system_list) for i in range(0, self.num_strata)
            ]

        # We need the index number to filter the stratum, so we use a for loop
        non_empty_stratum_index_list = combined_df['stratum_index'].value_counts().index.to_list()
        non_empty_stratum_index_list.sort()
        for i in range(0, self.num_strata):
            curr_stratum = self.strata[i]
            stratum_ind = non_empty_stratum_index_list[i]
            stratum_key_df = combined_df.loc[combined_df["stratum_index"] == stratum_ind, :]
            # Stratum method makes a deep copy of this method
            curr_stratum.construct_stratum_from_trials(
                stratum_key_df.loc[:, ["trial_id", "stratum_index", "key"]], stratum_ind)

        # Now store the key df with the stratum index, and push that information to the system
        self.key_df = combined_df.loc[:, ["trial_id", "stratum_index", "key"]]

        self.dirty_strata_cache()
        [system.add_stratum_index_to_system_data(combined_df.loc[:, ["trial_id", "stratum_index"]])
         for system in self.system_list]
        self.dirty_strata_cache()
        return self

    def aggregate_system_confidence_values(self):
        """
        Produces an aggregated system confidence value by taking the max of all of the confidence
        values, ensuring that if we stop that all systems are within the delta.

        Returns:
            float: An aggregated confidence value

        """
        delta = 0
        for conf_vals in self.system_confidence_list:
            delta_temp = conf_vals[2]
            if np.isnan(delta_temp):
                return np.nan
            if delta_temp > delta:
                delta = delta_temp
        return delta

    def aggregate_system_stats(self, sys_stat_list):
        """
        Aggregates the system stats list by taking the maximum of all system stats to get an
        aggregated value

        Args:
            sys_stat_list (list of float): The list of values, one value per system, to aggregate.

        Returns:
            float: An aggregated value.

        """
        return np.nanmax([stat for stat in sys_stat_list])


class StrataFirstSystemDecision(Strata):
    """
    Stratification Style that stratifies according to the first system but incorporates in the
    decisions as well as the system scores. Metrics with a threshold are required for this
    Strata class. It is recommended
    to specify the system ordering in aegis.data_processor.extract_files_from_directory() method
    so that the desired system is used.

    The stratification first stratifies all points by the system decision. Then, it gives
    num_strata/num_decisions bins for each decision, having a total of num_strata stratum.

    This provides both for better stratification as well as better statistical modeling, since if
    we were to model a metric as a sum of two binomials, stratifying by decision makes it so that
    in each strata one binomial is always 0, allowing for using one binomial to be a better
    approximation.
    """

    def __init__(self, num_strata, system_list):
        """
        Constructor to initialize strata. System objects are stored (by reference) in the strata.
        Calls superclass constructor and does nothing else.

        Insists that there are the same number of stratum in each decision, reducing the
        number of stratum if needed

        Args:
            num_strata (int): The number of strata to make
            system_list (list of :obj:`aegis.acteval.system.System`): The list of system objects.
        """
        # Call super() first because we need the df to get the combined df with the decisions
        super().__init__(num_strata, system_list)
        self.num_strata = num_strata
        self.stratify_for_pure_random = False
        if num_strata == -1:
            self.num_strata = 1
            self.stratify_for_pure_random = True
        self.num_systems = len(system_list)
        # Get the number of decisions
        combined_df = self.get_combined_systems_df()
        first_sys_id = self.system_list[0].system_id
        decision_values = combined_df[first_sys_id + '_dec'].unique().tolist()
        self.num_original_strata = num_strata
        # Now get a corrected number of strata
        if not self.stratify_for_pure_random:
            self.num_strata_per_decision = int(math.floor(self.num_strata / len(decision_values)))
            corr_num_strata = self.num_strata_per_decision*len(decision_values)
            # Correct the number of strata in case of a mistake
            self.num_strata = corr_num_strata
            self.num_original_strata = corr_num_strata
        # Called super() at the beginning

    def get_strata_alpha(self, alpha):
        """
        To avoid too many type one errors, get an alpha that is adjusted accordingly for
        multiple systems. This class ignores all but one system, so return just one system

        Args:
            alpha: (float) The original desired alpha

        Returns:
            float: new_alpha, The new desired alpha

        """
        new_alpha = alpha
        return new_alpha

    def stratify(self, bin_style='equal'):
        """
        Stratification method that stratifies purely by the first system and ignores
        other systems.

        Since this ignores the trial data, there is no trial_df argument.

        Start with the easiest strategy: stratify by score such that the bin widths (the score
        ranges, not the number of items in each bin, are the same), and stratify by the score of
        the first system.

        In this strata of stratum, the population size
        is the total number of trials in all of the stratum, not the total number of trials that
        have been sampled. For metrics such as recall where this cannot be computed directly,
        this will be estimated by by incorporating the estimates of each stratum in the
        stratified sampling manner.

        Args:
            bin_style (str): The style to stratify the bins. Default is 'equal'.
                'equal': Stratify the bins so that the range of values is equal, or the bins
                are of equal width
                'perc': Stratfiy by percentile, or so that an equal number of trials are in each bin

        Returns:
            aegis.acteval.strata.Strata: new_strata, the strata object after stratification

        """
        # We obtain the combined data frame from the systems
        combined_df = self.get_combined_systems_df()
        self.dirty_strata_cache()
        self.num_trials = combined_df.shape[0]

        # column 0 is 'trial_id', 1 is the first system
        first_sys_id = self.system_list[0].system_id
        # For now just stratify by the first system

        if self.stratify_for_pure_random:
            combined_df["stratum_index"] = 0
        else:
            decision_values = combined_df[first_sys_id + '_dec'].unique().tolist()
            num_decision_strata = int(math.floor(self.num_strata/len(decision_values)))
            combined_df["dec_stratum_index"] = -1
            combined_df["stratum_bin"] = -1
            for dec in decision_values:
                num_dec_values = len(combined_df.loc[combined_df[first_sys_id + '_dec'] == dec,
                                                     first_sys_id].unique())
                if num_dec_values > 1:
                    if bin_style == 'perc':
                        combined_df.loc[combined_df[first_sys_id + '_dec'] == dec,
                                        "stratum_bin"] = \
                            pd.qcut(
                                combined_df.loc[
                                    combined_df[first_sys_id + '_dec'] == dec, first_sys_id],
                                np.nanmin([num_dec_values, num_decision_strata]),
                                duplicates='drop'
                            )
                    else:
                        # Default:  cut bins into equal widths
                        combined_df.loc[combined_df[first_sys_id + '_dec'] == dec,
                                        "stratum_bin"] = \
                            pd.cut(
                                combined_df.loc[
                                    combined_df[first_sys_id + '_dec'] == dec, first_sys_id],
                                np.nanmin([num_dec_values, num_decision_strata]), duplicates='drop'
                            )
                elif num_dec_values == 1:
                    combined_df.loc[combined_df[first_sys_id + '_dec'] == dec,
                                    "stratum_bin"] = str(dec)
            combined_df["stratum_bin"] = combined_df["stratum_bin"].astype('category')
            combined_df["dec_stratum_index"] = combined_df["stratum_bin"].cat.codes
            combined_df["dec_stratum_index"] = combined_df["dec_stratum_index"].apply(str) + \
                '_' + combined_df[first_sys_id + '_dec'].apply(str)
            # This counts on the stratum_index mapping to be from 0 to (num_strata - 1)
            combined_df["stratum_index"] = combined_df["dec_stratum_index"].\
                astype('category').cat.codes

            combined_df.drop(columns=["stratum_bin"], inplace=True)
            combined_df.drop(columns=["dec_stratum_index"], inplace=True)

        combined_df["key"] = np.nan

        # Check if we need to construct fewer strata then specified
        # Rebuild stratum objects with fewer strata objects
        num_nonempty_strata = len(combined_df['stratum_index'].value_counts().index.to_list())
        # This covers any stratum differences
        if num_nonempty_strata != self.num_strata:
            self.num_strata = num_nonempty_strata
            self.strata = [
                aegis.acteval.stratum.Stratum(self.system_list) for i in range(0, self.num_strata)
            ]

        # We need the index number to filter the stratum, so we use a for loop
        non_empty_stratum_index_list = combined_df['stratum_index'].value_counts().index.to_list()
        non_empty_stratum_index_list.sort()
        for i in range(0, self.num_strata):
            curr_stratum = self.strata[i]
            stratum_ind = non_empty_stratum_index_list[i]
            stratum_key_df = combined_df.loc[combined_df["stratum_index"] == stratum_ind, :]
            # Stratum method makes a deep copy of this method
            curr_stratum.construct_stratum_from_trials(
                stratum_key_df.loc[:, ["trial_id", "stratum_index", "key"]], stratum_ind)

        # Now store the key df with the stratum index, and push that information to the system
        self.key_df = combined_df.loc[:, ["trial_id", "stratum_index", "key"]]

        [system.add_stratum_index_to_system_data(combined_df.loc[:, ["trial_id", "stratum_index"]])
         for system in self.system_list]
        self.dirty_strata_cache()
        return self

    def aggregate_system_confidence_values(self):
        """
        Produces an aggregated system confidence value by taking the confidence value of the first
        system and ignoring all other systems' values.

        Returns:
            float: An aggregated confidence value

        """
        delta = self.system_confidence_list[0][2]
        return delta

    def aggregate_system_stats(self, sys_stat_list):
        """
        Aggregates the system stats list by taking the first system's value. Ignores values
        of all other systems.

        Args:
            sys_stat_list (list of float): The list of values, one value per system, to aggregate.

        Returns:
            float: An aggregated value.

        """
        return sys_stat_list[0]


class StrataMultiSystemIntersectDecision(Strata):
    r"""
    Default Stratification Style that stratifies according to all of the systems and
    the decisions of each system. It creates
    :math:`\sqrt[num\_systems]{num\_strata}/2` per system per system decision
    to get as close as possible to handle floating point errors

    The system aggregates such that all systems have to be within their confidence for
    a successful round. Other stats are averaged.
    """

    def __init__(self, num_strata, system_list):
        r"""
        Constructor to initialize strata. System objects are stored (by reference) in the strata.
        Calls superclass constructor and does nothing else. It creates
        :math:`\sqrt[num\_systems]{num\_strata}` per system.

        Because of the possibility of there being a different number of decisions by system,
        this class will give reduced number of stratum in the stratify() method if need be.

        Args:
            num_strata (int): The number of strata to make
            system_list (list of :obj:`aegis.acteval.system.System`): The list of system objects.
        """
        self.num_systems = len(system_list)
        self.num_strata = num_strata
        self.stratify_for_pure_random = False
        if num_strata == -1:
            self.num_strata = 1
            self.stratify_for_pure_random = True
        num_strata_per_system = math.pow(self.num_strata, 1.0/len(system_list))
        self.num_strata_per_system = int(np.round(num_strata_per_system))
        # Correct the number of strata in case of a mistake or roundoff
        corr_num_strata = int(np.round(self.num_strata_per_system ** (len(system_list))))
        super().__init__(corr_num_strata, system_list)
        # This method does not handle reducing the stratum by the number of decisions:
        # that is done in post processing.
        self.stratify_for_pure_random = False
        if num_strata == -1:
            self.num_strata = 1
            self.stratify_for_pure_random = True

    def get_strata_alpha(self, alpha):
        """
        To avoid too many type one errors, get an alpha that is adjusted accordingly for
        multiple systems. This class uses all systems, so find an alpha whose union bound
        gives us the desired alpha.

        Args:
            alpha: (float) The original desired alpha

        Returns:
            float: new_alpha, The new desired alpha

        """
        new_alpha = 1 - np.power(1 - alpha, 1/self.num_systems)
        return new_alpha

    def stratify(self, bin_style='equal'):
        r"""
        Stratification method that stratifies purely by all of the systems. It creates
        :math:`\sqrt[num\_systems]{num\_strata}` strata for each system that are binned according
        to the specified bin style. Then it intersects the strata across each system to produce
        disjoint strata for sampling.

        Since this ignores the trial data, there is no trial_df argument.

        Args:
            bin_style (str): The style to stratify the bins. Default is 'equal'.
                'equal': Stratify the bins so that the range of values is equal, or the bins
                are of equal width
                'perc': Stratfiy by percentile, or so that an equal number of trials are in each bin

        Returns:
            aegis.acteval.strata.Strata: new_strata, the strata object after stratification

        """
        # We obtain the combined data frame from the systems
        combined_df = self.get_combined_systems_df()
        self.dirty_strata_cache()
        self.num_trials = combined_df.shape[0]

        # preserve ordering provided to us
        sys_id_list = [system.system_id for system in self.system_list]

        # First make a stratum_index for each system

        combined_df["stratum_index"] = 0
        if not self.stratify_for_pure_random:
            for sys_id in sys_id_list:
                sys_ind = sys_id_list.index(sys_id)
                decision_values = combined_df[str(sys_id) + '_dec'].unique().tolist()
                num_decision_strata = int(math.floor(self.num_strata_per_system /
                                                     len(decision_values)))
                combined_df[str(sys_id) + "_dec_stratum_index"] = -1
                combined_df[str(sys_id) + "_stratum_bin"] = -1
                for dec in decision_values:
                    num_dec_values = len(combined_df.loc[combined_df[str(sys_id) + '_dec'] == dec,
                                                         str(sys_id)].unique())
                    if num_dec_values > 1:
                        if bin_style == 'perc':
                            combined_df.loc[
                                combined_df[str(sys_id) + '_dec'] == dec,
                                str(sys_id) + "_stratum_bin"] = \
                                pd.qcut(
                                    combined_df.loc[
                                        combined_df[str(sys_id) + '_dec'] == dec, str(sys_id)],
                                    np.nanmin([num_dec_values, num_decision_strata]),
                                    duplicates='drop'
                                )
                        else:
                            # Default:  cut bins into equal widths
                            combined_df.loc[
                                combined_df[str(sys_id) + '_dec'] == dec,
                                str(sys_id) + "_stratum_bin"] = \
                                pd.cut(
                                    combined_df.loc[
                                        combined_df[str(sys_id) + '_dec'] == dec, str(sys_id)],
                                    np.nanmin([num_dec_values, num_decision_strata]),
                                    duplicates='drop'
                                )
                    elif num_dec_values == 1:
                        combined_df.loc[combined_df[str(sys_id) + '_dec'] == dec,
                                        "stratum_bin"] = str(dec)
                combined_df[str(sys_id) + "_stratum_bin"] = \
                    combined_df[str(sys_id) + "_stratum_bin"].astype('category')
                combined_df[str(sys_id) + "_dec_stratum_index"] = \
                    combined_df[str(sys_id) + "_stratum_bin"].cat.codes
                combined_df[str(sys_id) + "_dec_stratum_index"] = \
                    combined_df[str(sys_id) + "_dec_stratum_index"].apply(str) + \
                    '_' + combined_df[str(sys_id) + '_dec'].apply(str)
                # This counts on the stratum_index mapping to be from 0 to (num_strata - 1)
                combined_df[str(sys_id) + "_stratum_index"] = \
                    combined_df[str(sys_id) + "_dec_stratum_index"].astype('category').cat.codes
                combined_df["stratum_index"] += (self.num_strata_per_system**sys_ind) * \
                                                (combined_df[str(sys_id) + "_stratum_index"])
                combined_df.drop(columns=[str(sys_id) + "_stratum_bin"], inplace=True)
                combined_df.drop(columns=[str(sys_id) + "_dec_stratum_index"], inplace=True)
                combined_df.drop(columns=[str(sys_id) + "_stratum_index"], inplace=True)

        combined_df["key"] = np.nan

        # Check if we need to construct fewer strata then specified
        # Rebuild stratum objects with fewer strata objects
        num_nonempty_strata = len(combined_df['stratum_index'].value_counts().index.to_list())
        # This covers any stratum differences
        if num_nonempty_strata != self.num_strata:
            self.num_strata = num_nonempty_strata
            self.strata = [
                aegis.acteval.stratum.Stratum(self.system_list) for i in range(0, self.num_strata)
            ]

        # We need the index number to filter the stratum, so we use a for loop
        non_empty_stratum_index_list = combined_df['stratum_index'].value_counts().index.to_list()
        non_empty_stratum_index_list.sort()
        for i in range(0, self.num_strata):
            curr_stratum = self.strata[i]
            stratum_ind = non_empty_stratum_index_list[i]
            stratum_key_df = combined_df.loc[combined_df["stratum_index"] == stratum_ind, :]
            # Stratum method makes a deep copy of this method
            curr_stratum.construct_stratum_from_trials(
                stratum_key_df.loc[:, ["trial_id", "stratum_index", "key"]], stratum_ind)

        # Now store the key df with the stratum index, and push that information to the system
        self.key_df = combined_df.loc[:, ["trial_id", "stratum_index", "key"]]

        [system.add_stratum_index_to_system_data(combined_df.loc[:, ["trial_id", "stratum_index"]])
         for system in self.system_list]
        self.dirty_strata_cache()
        return self

    def aggregate_system_confidence_values(self):
        """
        Produces an aggregated system confidence value by taking the max of all of the confidence
        values, ensuring that if we stop that all systems are within the delta.

        Returns:
            float: An aggregated confidence value

        """
        delta = 0
        for conf_vals in self.system_confidence_list:
            delta_temp = conf_vals[2]
            if np.isnan(delta_temp):
                return np.nan
            if delta_temp > delta:
                delta = delta_temp
        return delta

    def aggregate_system_stats(self, sys_stat_list):
        """
        Aggregates the system stats list by taking the maximum of all system stats to get an
        aggregated value

        Args:
            sys_stat_list (list of float): The list of values, one value per system, to aggregate.

        Returns:
            float: An aggregated value.

        """
        return np.nanmax([stat for stat in sys_stat_list])
