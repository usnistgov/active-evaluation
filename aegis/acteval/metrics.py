import abc
import pandas as pd
import scipy.stats
import numpy as np
import sklearn.metrics as skm


class Metric(abc.ABC):
    """
    Abstract class representing a metric, which both scores trials as well as esimates the means,
    variances, and confidence intervals of a set of trials.

    Metric is an abstract class that must be implemented. Given a stratum, compute the mean,
    variance, and confidence intervals. Given strata, compute the
    overall mean, variance,  and confidence intervals. Method implementations common to all
    metrics are implemented here.

    """

    def __init__(self):
        """
        Constructor.

        Args:
            No arguments.
        """
        super().__init__()

    @abc.abstractmethod
    def get_metric_name(self):
        """
        Returns the name of the metric as a string.

        Returns:
            str: the metric name

        """
        pass

    @abc.abstractmethod
    def find_needed_initial_samples(self, stratum, initial_samples_per_bin, rng):
        """
        In order to cover the case where we don't sample from each stratum necessary edge cases
        (Such as when the metric is a combination of two components) and to ensure samples
        from each stratum, this method will find such examples and will sample uniformly at random

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum object to find
                needed samples
            initial_samples_per_bin (int): The number of initial samples per bin
            rng (:obj:`numpy.random.RandomState`): random state object with stored random state

        Returns:
            list of object: A list of trial_id values to sample

        """
        pass

    @abc.abstractmethod
    def estimate_samples_all_systems(self, stratum):
        """
        Gets the number of sampled trials relevant to the metric for each system.
        Neither Stratum nor System objects are not updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum to determine how
                many samples count towards the population

        Returns:
            list of int: A list of samples counted for the systems, in the order of the
            systems in the stratum object.

        """
        pass

    @abc.abstractmethod
    def estimate_pop_all_systems(self, stratum):
        """
        Gives the total number of trials to be considered for the metric computation.

        Although it seems obvious for metrics like "accuracy" where the population is the
        number of trials, it can be more complex for other methods. For instance, when computing
        precision, the population is the number of trials where the output converts to a positive.
        This means that the population is far less than the number of trials.
        Neither Stratum nor System objects are not updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum of which to determine the
                population.

        Returns:
            list of float: A list of population sizes, with the ith element being the population
            size for system i. Population sizes can differ for systems for metrics like precision,
            where the population size is the number of system positive trials. Although true
            populations are always integers, estimates could be either ints or floats.

        """
        pass

    def estimate_pop_frac_all_systems(self, stratum):
        """
        Estimate the fraction of relevant trials for each stratum for each system. Calls
        estimate_pop_all_systems() method of metric class. Each fraction is a value from 0 to 1.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum object

        Returns:
            list of float: A list of population percentages, with the ith element being the
            population percentage estimate for system i. For some metrics this is known; for
            other metrics this is estimated.

        """
        pop_perc_list = self.estimate_pop_all_systems(stratum)/stratum.num_trials
        return pop_perc_list

    @abc.abstractmethod
    def get_trials_to_sample_from(self, stratum):
        """
        Given a stratum, determine the subset of trials that the stratum should draw from.
        If all samples are relevant, return all of the trial ids. This is useful for metrics like
        precision where we must subset the number of trials so that we only sample from trials
        where at least one system gives that trial a positive decision.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum to identify which samples
            are relevant for sampling

        Returns:
            list of object: A list of trial ids to sample from

        """
        pass

    @abc.abstractmethod
    def estimate_score_all_systems(self, stratum):
        """
        Gives the scores for each system on the samples obtained so far.
        Neither Stratum nor System objects are not updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum to score.

        Returns:
            list of float: A list of scores, one score per system

        """
        pass

    @abc.abstractmethod
    def estimate_score_variance_all_systems(self, stratum):
        """
        Gives the score_variances for each system on the samples obtained so far.
        Neither Stratum nor System objects are not updated during this computation. The score
        variance is also enabled at the stratum level to allow for metric corrections such as
        priors for a bayesian confidence interval.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum of which to estimate
                the score variance.

        Returns:
            list of float: A list of score variances, with one score variance per system
            ordered by the order of systems in the system id list.

        """
        pass

    def estimate_score_variance_upper_all_systems(self, stratum):
        """
        Gives the score_variances for each system on the samples obtained so far with
        upper bound estimates on the population.
        Neither Stratum nor System objects are not updated during this computation. The score
        variance is also enabled at the stratum level to allow for metric corrections such as
        priors for a bayesian confidence interval.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum of which to estimate the
            score variance.

        Returns:
            list of float: A list of score variances for upper populations,
            with one score variance per system ordered by the order of systems in the
            system id list.

        """
        sys_score_var_list = self.estimate_score_variance_all_systems(stratum)
        sys_pop_list = self.estimate_pop_all_systems(stratum)
        sys_pop_variance_list = self.estimate_pop_variance_all_systems(stratum)
        sys_score_var_upper_list = [0 for i in range(0, len(stratum.system_list))]
        for ind in range(0, len(stratum.system_list)):
            if sys_pop_list[ind] != 0:
                sys_score_var_upper_list[ind] = sys_score_var_list[ind]
            elif sys_pop_variance_list[ind] == 0:
                sys_score_var_upper_list[ind] = np.nan
            else:
                # Here, we know that have not obtained a single relevant sample but are
                # not certain that there is a relevant sample
                # So we assume that the score can be anything (0–1) but that its contribution
                # is mitigated by the population.
                # We give the highest variance possible for a proportion, which is 0.25
                # to balance the unknown and to insist that we get a finite number for estimation
                sys_score_var_upper_list[ind] = 0.25
                # Now give this a finite population correction
                sampled_trials = stratum.get_combined_systems_score_df().shape[0]
                total_trials = stratum.num_trials
                fpc = 0
                if total_trials > 1:
                    fpc = 1 - ((sampled_trials - 1) / (total_trials - 1))
                sys_score_var_upper_list[ind] = sys_score_var_upper_list[ind] * fpc

        return sys_score_var_upper_list

    def estimate_pop_variance_all_systems(self, stratum):
        """
        Estimate the uncertainty (variance) of our estimate of the population size. This is 0
        when the population is known.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum of which to estimate
            the population variance.

        Returns:
            list of float: A list of population variances, with one population variance per system,
            ordered by the order of systems int he system id list.

        """
        pop_var_list = self.estimate_pop_frac_variance_all_systems(stratum)
        pop_var_list = [pv * stratum.num_trials for pv in pop_var_list]
        return pop_var_list

    @abc.abstractmethod
    def estimate_pop_frac_variance_all_systems(self, stratum):
        """
        Estimate the uncertainty (variance) of our estimate of the population fraction. Returns
        0 when the population is known.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum of which to estimate
                the population fraction variance.

        Returns:
            list of float: A list of population fraction variances, with one estimate
            per system,
            ordered by the order of systems in the system id list.

        """

    def estimate_population_intervals_all_systems(self, stratum, alpha):
        r"""
        Get the confidence intervals for all systems' populations,
        represented as a three-element list
        [lower, upper, delta] for each system on a given stratum

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum object
            alpha (float): the level to return the 1 - alpha population confidence for

        Returns:
            list of list of float: A list of lists, where there is one list per system, and each
            system is a
            three-element list [lower, upper, delta], where delta is the higher of
            (population_est - lower) and (higher - population)

        """
        system_pop_conf_values = [[0, 0, 0] for i in range(0, len(stratum.system_list))]
        system_pops = self.estimate_pop_all_systems(stratum)
        system_pop_frac_vars = self.estimate_pop_frac_variance_all_systems(stratum)
        for ind in range(0, len(stratum.system_list)):
            se = np.sqrt(system_pop_frac_vars[ind])
            pop = system_pops[ind]
            if np.isnan(pop):
                system_pop_conf_values[ind] = [np.nan, np.nan, np.nan]
                continue
            z = scipy.stats.norm.ppf(1 - (alpha / 2), loc=0, scale=1)
            delta = z * se * stratum.num_trials
            lower = pop - delta
            upper = pop + delta
            system_pop_conf_values[ind] = [lower, upper, delta]

        return system_pop_conf_values

    def estimate_population_intervals_all_systems_strata(self, strata, alpha):
        r"""
        Get the confidence intervals for all systems' populations,
        represented as a three-element list
        [lower, upper, delta] for each system

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object
            alpha (float): the level to return the 1 - alpha population confidence for

        Returns:
            list of list of float: A list of lists, where there is one list per system, and each
            system is a
            three-element list [lower, upper, delta], where delta is the higher of
            (population_est - lower) and (higher - population)

        """
        # system_pop_score_variances = strata.estimate_pop_score_variance_all_systems(self)
        system_pop_conf_values = [[0, 0, 0] for i in range(0, len(strata.system_list))]

        for ind in range(0, len(strata.system_list)):
            system = strata.system_list[ind]
            se = np.sqrt(system.population_frac_variance)
            pop = system.population
            if np.isnan(pop):
                system_pop_conf_values[ind] = [np.nan, np.nan, np.nan]
                continue
            z = scipy.stats.norm.ppf(1 - (alpha / 2), loc=0, scale=1)
            delta = z * se * strata.num_trials
            lower = pop - delta
            upper = pop + delta
            system_pop_conf_values[ind] = [lower, upper, delta]

        return system_pop_conf_values

    def get_confidence_intervals_true_pop_all_systems(self, strata, alpha):
        r"""
        Get the confidence intervals for all system without uncertainties
        of population estimates, represented as a three-element list
        [lower, upper, delta] for each system

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object
            alpha (float): the level to return the 1 - alpha confidence for

        Returns:
            list of lsit of float: A list of lists, where there is one list per system,
            and each system is a
            three-element list [lower, upper, delta], where delta is the higher of
            (score_est - lower)
            and (higher - score_est)

        """
        # system_scores = strata.estimate_score_all_systems(self)
        # system_score_variances = strata.estimate_score_variance_all_systems(self)
        # We have three bounds:
        # 1) The bound computed from stratified sampling assuming our point estimates of population
        # are 100% correct and our metric estimate varies
        # 2) The bound computed from
        system_conf_values = [[0, 0, 0] for i in range(0, len(strata.system_list))]
        for ind in range(0, len(strata.system_list)):
            system = strata.system_list[ind]
            se = np.sqrt(system.score_variance_upper)
            # If we have no upper variance, use the regular score variance
            if system.score_variance_upper == 0:
                se = np.sqrt(system.score_variance)
            score = system.score
            if np.isnan(score):
                system_conf_values[ind] = [np.nan, np.nan, np.nan]
                continue
            # n for t dist: n = system.sampled_trials
            # t dist: t = scipy.stats.t.ppf(1 - (alpha / 2), n - 1, loc=0, scale=1)
            z = scipy.stats.norm.ppf(1 - (alpha / 2), loc=0, scale=1)
            delta = z * se
            lower = score - delta
            upper = score + delta
            delta_adj = delta
            system_conf_values[ind] = [lower, upper, delta_adj]

        return system_conf_values

    def get_confidence_intervals_all_systems(self, strata, alpha):
        r"""
        Get the confidence intervals for all systems, represented as a three-element list
        [lower, upper, delta] for each system. These confidence intervals include uncertainty
        for uncertainty in the population estimates.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object
            alpha (float): the level to return the 1 - alpha confidence for

        Returns:
            list of list of float: A list of lists, where there is one list per system, and each
            system is a three-element list [lower, upper, delta], where delta is the
            higher of (score_est - lower) and (higher - score_est)

        """
        # system_scores = strata.estimate_score_all_systems(self)
        system_conf_values = [[0, 0, 0] for i in range(0, len(strata.system_list))]
        for ind in range(0, len(strata.system_list)):
            system = strata.system_list[ind]
            # Assume two variances are independent
            se = np.sqrt(system.score_variance_upper + system.population_frac_variance)
            # se = np.sqrt(system.score_variance_upper + system.population_frac_variance +
            #                              np.sqrt(system.score_variance_upper) *
            #                              np.sqrt(system.population_frac_variance))
            # If we have only one strata, we do not consider the population variance
            # since the population variance concerns the uncertainty in the weighting
            if strata.num_strata == 1:
                se = np.sqrt(system.score_variance_upper)
                # If we have no upper variance, use the regular score variance
                if system.score_variance_upper == 0:
                    se = np.sqrt(system.score_variance)
            # If we have no upper variance, use the regular score variance
            elif system.score_variance_upper == 0:
                se = np.sqrt(system.score_variance + system.population_frac_variance)
            score = system.score
            if np.isnan(score):
                system_conf_values[ind] = [np.nan, np.nan, np.nan]
                continue
            # n for t dist: n = system.sampled_trials
            # t dist: t = scipy.stats.t.ppf(1 - (alpha / 2), n-1, loc=0, scale=1)
            z = scipy.stats.norm.ppf(1 - (alpha / 2), loc=0, scale=1)
            delta = z * se
            # In theory, score_lower <= score and score_upper >= score, but add this
            # check to guarantee that we do not make the interval smaller
            lower = score - delta
            upper = score + delta
            delta_adj = max(score - lower, upper - score)
            system.confidence_value = delta_adj
            system_conf_values[ind] = [lower, upper, delta_adj]

        return system_conf_values

    @abc.abstractmethod
    def get_actual_score(self, system_df, key_df):
        """
        Using a full system data frame and a key data frame, compute the score. This method is
        used when by the OracleScript that has a full key df.

        Args:
            system_df (:obj:`pandas.core.frame.DataFrame`): The data frame of a single system.
                For metrics that have thresholds,
                it is assumed that this system has 'score' or 'decision' columns as needed.
            key_df (:obj:`pandas.core.frame.DataFrame`): The answer key data frame.

        Returns:
            float: A score for that system according to that key.

        """


class BinaryPrecisionMetric(Metric):
    """
    Instantiation of a Metric corresponding to the simplest metric "precision", or true prositives
    over total true positives and false positives.
    Only for binary classification but it handles thresholds and arbitrary key values.

    This metric approximates accuracy as a combination of two proportions. The first proportion
    is the proportion of scores that the system classifies as true. The second proportion is
    of scores that the system classifies as false. We use the system scores rather than the ground
    truth to split the proportions because we know all of teh system scores.

    With a threshold, it considers all of the values greater than the threshold to be in the
    high_key_value class with the remainder of examples to be in the low_key_value_class
    """

    low_key_value = 0
    high_key_value = 1

    def __init__(self, key_values=None, score_prior_samples_correction=0.5):
        """
        Constructor.

        Args:
            key_values (list, optional): The two key values to use, specified as [low, high].
                None by default, which is replaced with [0, 1].
            score_prior_samples_correction (float, optional): the adjustment on the score to
                correct the point estimate
                for proportions near 0 or 1. In computations, this number is divided by the
                square root of the number of samples. Defaults to 0.5.
        """
        if key_values is None:
            key_values = [0, 1]
        self.low_key_value = key_values[0]
        self.high_key_value = key_values[1]
        self.prior_samples_correction = score_prior_samples_correction
        super().__init__()

    def get_metric_name(self):
        """
        Returns the name of the metric as a string.

        Returns:
            str: the metric name

        """
        return "BinaryPrecisionMetric"

    def convert_thresholds_to_decisions(self, system):
        """
        When computing Precision, we need to use the threshold specified to convert. This
        assumes a binary classification problem. This converts to the high key value if the score
        is strictly higher than the threshold.

        The decision is added to the System data frame in the column "decision". The score
        column is preserved.

        Args:
            system (:obj:`aegis.acteval.system.System`): the system containing the scores

        Returns:
            aegis.acteval.system.System: The modified System object with a modified system_df with
            decisions in the "decision" column as well as preserving the score values.

        """

        # Important that we copy by reference
        sys_id = system.system_id
        sys_df = system.system_df
        # For now, use the first threshold value regardless of name
        threshold_df = system.threshold_df
        sys_threshold_val = threshold_df.loc[
            threshold_df["system_id"] == sys_id, "value"
        ].iloc[0]
        sys_df["decision"] = sys_df["score"] > sys_threshold_val
        sys_df["decision"] = sys_df["decision"].map({False: self.low_key_value,
                                                     True: self.high_key_value})
        return system

    def find_needed_initial_samples(self, stratum, initial_samples_per_bin, rng):
        """
        In order to cover the case where we don't sample from each stratum necessary edge cases
        (Such as when the metric is a combination of two components) and to ensure samples
        from each stratum, this method will find such examples and will sample uniformly at random

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum object to find
                needed samples
            initial_samples_per_bin (int): The number of initial samples per bin
            rng (:obj:`numpy.random.RandomState`): random state object with stored random state

        Returns:
            list of object: A list of trial_id values to sample

        """
        if stratum.num_trials == 0:
            # No samples needed from an empty stratum
            return []

        combined_df = stratum.get_combined_systems_df()

        # Perform this for each system and split samples so that so many are relevant
        # for each system
        samples_per_system = int(np.floor(initial_samples_per_bin/len(stratum.system_list)))
        sys_remaining_samples = initial_samples_per_bin
        samples_list = []
        for ind in range(0, len(stratum.system_list)):
            sys_id = stratum.system_list[ind].system_id
            sys_decisions = (combined_df[str(sys_id) + "_dec"] == self.high_key_value)
            sys_non_score_df = combined_df.loc[(pd.isna(combined_df["key"]) & sys_decisions), :]
            if sys_non_score_df.shape[0] < samples_per_system:
                curr_samples = sys_non_score_df.sample(n=sys_non_score_df.shape[0], replace=False,
                                                       random_state=rng)
            else:
                curr_samples = sys_non_score_df.sample(n=samples_per_system, replace=False,
                                                       random_state=rng)
            samples_list.extend(curr_samples["trial_id"])
            sys_remaining_samples -= len(curr_samples)

        # Sample the rest from what is left out of the total population
        valid_decisions = pd.Series([False] * combined_df.shape[0], index=combined_df.index)
        for ind in range(0, len(stratum.system_list)):
            sys_id = stratum.system_list[ind].system_id
            valid_decisions = (valid_decisions | (combined_df[str(sys_id) + "_dec"]
                                                  == self.high_key_value))

        score_df = combined_df.loc[(pd.notna(combined_df["key"]) & valid_decisions), :]
        non_score_df = combined_df.loc[(pd.isna(combined_df["key"]) & valid_decisions), :]
        num_needed_samples = sys_remaining_samples
        remaining_samples = num_needed_samples - score_df.shape[0]
        if remaining_samples <= 0:
            return sorted(samples_list)

        if non_score_df.shape[0] < remaining_samples:
            remaining_samples = non_score_df.shape[0]

        new_sample = non_score_df.sample(n=remaining_samples, replace=False, random_state=rng)
        samples_list.extend(new_sample["trial_id"])
        return sorted(samples_list)

    def estimate_samples_all_systems(self, stratum):
        """
        Gets the number of sampled trials relevant to the metric for each system. For Binary
        Classification Precision, only trials where the system decides on the high key value count.
        Neither Stratum nor System objects are not updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum to determine how
                many samples count towards the population

        Returns:
            list of int: A list of samples counted for the systems, in the order of the
            systems in the stratum object.

        """
        if stratum.num_trials == 0:
            return [0 for i in range(0, len(stratum.system_list))]

        score_df = stratum.get_combined_systems_score_df()
        if score_df.shape[0] == 0:
            return[0 for i in range(0, len(stratum.system_list))]

        sys_sample_list = [
            sum(score_df[str(stratum.system_list[i].system_id) + '_dec'] == self.high_key_value)
            for i in range(0, len(stratum.system_list))]
        return sys_sample_list

    def estimate_pop_all_systems(self, stratum):
        """
        Gets the number of sampled trials relevant to the metric for each system. For Binary
        Precision, the population is known and is the number of trials where each system has
        a decision of the high key value.
        Neither Stratum nor System objects are not updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum of which to determine the
                population.

        Returns:
            list of int: A list of population sizes, with the ith element being the population
            size for system i.

        """
        if stratum.num_trials == 0:
            return [0 for i in range(0, len(stratum.system_list))]

        score_df = stratum.get_combined_systems_df()

        if score_df.shape[0] == 0:
            return[0 for i in range(0, len(stratum.system_list))]

        sys_pop_list = [
            sum(score_df[stratum.system_list[i].system_id + '_dec']
                == self.high_key_value)
            for i in range(0, len(stratum.system_list))
        ]
        return sys_pop_list

    def estimate_pop_frac_variance_all_systems(self, stratum):
        """
        Estimate the uncertainty (variance) of our estimate of the population fraction. Returns
        0 when the population is known. The population is known for Precision so 0 is returned.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum of which to estimate
                the population fraction variance.

        Returns:
            list of float: A list of population fraction variances, with one estimate
            per system,
            ordered by the order of systems in the system id list.

        """
        sys_pop_var_list = [0 for i in range(0, len(stratum.system_list))]
        return sys_pop_var_list

    def estimate_score_variance_all_systems(self, stratum):
        """
        Gives the score_variances for each system on the samples obtained so far.
        Neither Stratum nor System objects are not updated during this computation. The score
        variance is also enabled at the stratum level to allow for metric corrections such as
        priors for a bayesian confidence interval.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum of which to estimate
                the score variance.

        Returns:
            list of float: A list of score variances, with one score variance per system
            ordered by the order of systems in the system id list.

        """
        # If no trials in stratum, return nan
        if stratum.num_trials == 0:
            return [np.nan for system in stratum.system_list]

        score_df = stratum.get_combined_systems_score_df()
        # If no trials in key, return nan
        if score_df.shape[0] == 0:
            return [np.nan for system in stratum.system_list]

        samples_list = self.estimate_samples_all_systems(stratum)

        var_list = [0 for i in range(0, len(stratum.system_list))]
        for ind in range(0, len(stratum.system_list)):
            if samples_list[ind] == 0:
                var_list[ind] = np.nan
            else:
                sys_id = stratum.system_list[ind].system_id
                tp_df = score_df.loc[score_df[str(sys_id) + "_dec"] == self.high_key_value]
                tp = sum(tp_df[str(sys_id) + "_dec"] == tp_df["key"])
                fp = samples_list[ind] - tp
                alpha = tp + self.prior_samples_correction
                beta = fp + self.prior_samples_correction
                var_list[ind] = (alpha * beta) / (pow((alpha + beta), 2) * (alpha+beta+1))
        return var_list

    def estimate_score_all_systems(self, stratum):
        """
        Gives the scores for each system on the samples obtained so far.
        Neither Stratum nor System objects are not updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum to score.

        Returns:
            list of float: A list of scores, one score per system

        """
        # If no trials in stratum, return nan
        if stratum.num_trials == 0:
            return [np.nan for system in stratum.system_list]

        score_df = stratum.get_combined_systems_score_df()
        # If no trials in key, return nan
        if score_df.shape[0] == 0:
            return [np.nan for system in stratum.system_list]
        # score_sample_adj = self.prior_samples_correction
        samples_list = self.estimate_samples_all_systems(stratum)
        pop_list = self.estimate_pop_all_systems(stratum)
        score_list = [0 for i in range(0, len(stratum.system_list))]
        for ind in range(0, len(stratum.system_list)):
            sys_id = stratum.system_list[ind].system_id
            if samples_list[ind] == 0:
                score_list[ind] = np.nan
            else:
                tp_df = score_df.loc[score_df[str(sys_id) + "_dec"] == self.high_key_value]
                tp = sum(tp_df[str(sys_id)+"_dec"] == tp_df["key"])
                num_samples = samples_list[ind]
                sys_pop = pop_list[ind]
                fpc = 0
                if sys_pop > 1:
                    fpc = 1 - ((num_samples - 1) / (sys_pop - 1))
                score_list[ind] = (tp + self.prior_samples_correction*fpc) / \
                    (samples_list[ind] + 2 * self.prior_samples_correction * fpc)
        return score_list

    def get_actual_score(self, system_df, key_df):
        """
        Using a full system data frame and a key data frame, compute the score. This method is
        used when by the OracleScript that has a full key df. This computes the system precision
        on the system. This assumes that the
        :func:`aegis.acteval.metrics.Metric.convert_thresholds_to_decision()` method has been
        called to produce the system_df

        Args:
            system_df (:obj:`pandas.core.frame.DataFrame`): The data frame of a single system.
                For metrics that have thresholds,
                it is assumed that this system has 'score' or 'decision' columns as needed.
            key_df (:obj:`pandas.core.frame.DataFrame`): The answer key data frame.

        Returns:
            float: A score for that system according to that key.

        """
        score_df = pd.merge(key_df, system_df, on="trial_id", how="left")
        system_precision = skm.precision_score(score_df["key"], score_df["decision"],
                                               pos_label=self.high_key_value)
        return system_precision

    def get_trials_to_sample_from(self, stratum):
        """
        Given a stratum, determine the subset of trials that the stratum should draw from.
        If all samples are relevant, return all of the trial ids. For Precision,
        this is all trials where some system has a decision of the high key value.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum to identify which samples
            are relevant for sampling

        Returns:
            list of object: A list of trial ids to sample from

        """
        score_df = stratum.get_combined_systems_df()

        valid_decisions = pd.Series([False] * score_df.shape[0], index=score_df.index)

        for ind in range(0, len(stratum.system_list)):
            sys_id = stratum.system_list[ind].system_id
            valid_decisions = valid_decisions | \
                (score_df[str(sys_id) + "_dec"] == self.high_key_value)

        unsampled_trials = score_df.loc[pd.isna(score_df["key"])
                                        & valid_decisions, "trial_id"]
        return unsampled_trials


class BinaryRecallMetric(Metric):
    """
    Instantiation of a Metric corresponding to the simplest metric "Recall", or true prositives
    over total true positives and false positives.
    Only for binary classification but it handles thresholds and arbitrary key values.

    This metric approximates accuracy as a combination of two proportions. The first proportion
    is the proportion of scores that the system classifies as true. The second proportion is
    of scores that the system classifies as false. We use the system scores rather than the ground
    truth to split the proportions because we know all of teh system scores.

    With a threshold, it considers all of the values greater than the threshold to be in the
    high_key_value class with the remainder of examples to be in the low_key_value_class
    """

    low_key_value = 0
    high_key_value = 1

    def __init__(self, key_values=None, score_prior_samples_correction=0.5,
                 pop_samples_correction=0.5):
        """
        Constructor.

        Args:
            key_values (list, optional): The two key values to use, specified as [low, high].
                None by default, which is replaced with [0, 1].
            score_samples_correction (float, optional): the adjustment on the score to correct the
                point estimate
                for proportions near 0 or 1. In computations, this number is divided by the
                square root of the number of samples. Defaults to 0.5.
            pop_samples_correction (float, optional): The adjustment to the population prior for
                estimating
                upper population estimates
        """
        if key_values is None:
            key_values = [0, 1]
        self.low_key_value = key_values[0]
        self.high_key_value = key_values[1]
        self.prior_samples_correction = score_prior_samples_correction
        self.pop_samples_correction = pop_samples_correction
        super().__init__()

    def get_metric_name(self):
        """
        Returns the name of the metric as a string.

        Returns:
            str: the metric name

        """
        return "BinaryRecallMetric"

    def convert_thresholds_to_decisions(self, system):
        """
        When computing Recall, we need to use the threshold specified to convert. This
        assumes a binary classification problem. This converts to the high key value if the score
        is strictly higher than the threshold.

        The decision is added to the System data frame in the column "decision". The score
        column is preserved.

        Args:
            system (:obj:`aegis.acteval.system.System`): the system containing the scores

        Returns:
            aegis.acteval.system.System: The modified System object with a modified system_df with
            decisions in the "decision" column as well as preserving the score values.

        """

        # Important that we copy by reference
        sys_id = system.system_id
        sys_df = system.system_df
        # For now, use the first threshold value regardless of name
        threshold_df = system.threshold_df
        sys_threshold_val = threshold_df.loc[
            threshold_df["system_id"] == sys_id, "value"
        ].iloc[0]
        sys_df["decision"] = sys_df["score"] > sys_threshold_val
        sys_df["decision"] = sys_df["decision"].map({False: self.low_key_value,
                                                     True: self.high_key_value})
        return system

    def find_needed_initial_samples(self, stratum, initial_samples_per_bin, rng):
        """
        In order to cover the case where we don't sample from each stratum necessary edge cases
        (Such as when the metric is a combination of two components) and to ensure samples
        from each stratum, this method will find such examples and will sample uniformly at random

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum object to find
                needed samples
            initial_samples_per_bin (int): The number of initial samples per bin
            rng (:obj:`numpy.random.RandomState`): random state object with stored random state

        Returns:
            list of object: A list of trial_id values to sample

        """
        if stratum.num_trials == 0:
            # No samples needed from an empty stratum
            return []

        combined_df = stratum.get_combined_systems_df()
        score_df = combined_df.loc[pd.notna(combined_df["key"]), :]
        non_score_df = combined_df.loc[pd.isna(combined_df["key"]), :]
        num_needed_samples = initial_samples_per_bin
        remaining_samples = num_needed_samples - score_df.shape[0]
        if remaining_samples <= 0:
            return []

        if non_score_df.shape[0] < remaining_samples:
            remaining_samples = non_score_df.shape[0]

        new_sample = non_score_df.sample(n=remaining_samples, replace=False, random_state=rng)
        return sorted(list(new_sample["trial_id"]))

    def estimate_samples_all_systems(self, stratum):
        """
        Gets the number of sampled trials relevant to the metric for each system. For Recall,
        only trials with the key high key value will count, so this returns the number of
        sampled trials whose key is the high key value.
        Neither Stratum nor System objects are not updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum to determine how
                many samples count towards the population

        Returns:
            list of int: A list of samples counted for the systems, in the order of the
            systems in the stratum object.

        """
        if stratum.num_trials == 0:
            return [0 for i in range(0, len(stratum.system_list))]

        score_df = stratum.get_combined_systems_score_df()
        score_df = score_df[score_df['key'] == self.high_key_value]
        sys_sample_list = [
            score_df.shape[0] for i in range(0, len(stratum.system_list))
        ]
        return sys_sample_list

    def estimate_pop_all_systems(self, stratum):
        """
        Gets the number of sampled trials relevant to the metric for each system. For Recall
        We are unable to calculate this so we can estimate it by looking at the proportion of
        keys that are high_key_value to the overall samples and extrapolate this to the population.
        Neither Stratum nor System objects are not updated during this computation.


        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum of which to determine the
                population.

        Returns:
            list of float: A list of population size estimates, with the ith element being the
            population estimate for system i. Values are floats because these values are estimates.

        """
        if stratum.num_trials == 0:
            return [0 for i in range(0, len(stratum.system_list))]

        score_df = stratum.get_combined_systems_score_df()
        score_key_pop = score_df.shape[0]

        if score_df.shape[0] == 0:
            return[0 for i in range(0, len(stratum.system_list))]

        score_key_valid_samp = score_df[score_df['key'] == self.high_key_value].shape[0]
        pop_prop = score_key_valid_samp/score_key_pop

        stratum.population_size = stratum.num_trials

        sys_pop_list = [
            int(np.floor(stratum.population_size*pop_prop))
            for i in range(0, len(stratum.system_list))
        ]
        return sys_pop_list

    def estimate_pop_frac_variance_all_systems(self, stratum):
        """
        Estimate the uncertainty (variance) of our estimate of the population fraction. Returns
        0 when the population is known. For Recall, the true population is not known so this
        is almost always nonzero.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum of which to estimate
                the population fraction variance.

        Returns:
            list of float: A list of population fraction variances, with one estimate
            per system,
            ordered by the order of systems in the system id list.

        """

        # If we have only one stratum, we set this to 0; this keeps the random sampler unchanged

        sys_samples_list = self.estimate_samples_all_systems(stratum)
        # We need the total number of relevant and non-relevant sampled trials
        score_df = stratum.get_combined_systems_score_df()
        sys_pop_var_list = [0 for i in range(0, len(stratum.system_list))]
        for ind in range(0, len(stratum.system_list)):
            # Even if the population is not zero, we have a variance
            if score_df.shape[0] == 0:
                # We need a trial, even if not relevant to estimate the variance
                sys_pop_var_list[ind] = np.nan
            else:
                sys_trials = score_df.shape[0]
                sys_samples = sys_samples_list[ind]
                # Correction to have a prior
                alpha_v = sys_trials + self.pop_samples_correction
                beta_v = sys_samples + self.pop_samples_correction
                # Use Beta distribution variance
                sys_pop_var_list[ind] = (alpha_v * beta_v) / (pow((alpha_v + beta_v), 2) *
                                                              (alpha_v + beta_v + 1))
                sys_pop_var_list[ind] = sys_pop_var_list[ind]
        return sys_pop_var_list

    def get_trials_to_sample_from(self, stratum):
        """
        Given a stratum, determine the subset of trials that the stratum should draw from.
        If all samples are relevant, return all of the trial ids. For recall, this is any
        unsampled trial; even though we do not know which ones are relevant, we have no way
        of rejecting a specific trial before sampling.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum to identify which samples
            are relevant for sampling

        Returns:
            list of object: A list of trial ids to sample from

        """
        unsampled_trials = stratum.stratum_key_df.loc[pd.isna(stratum.stratum_key_df["key"]),
                                                      "trial_id"]
        return unsampled_trials

    def estimate_score_all_systems(self, stratum):
        """
        Gives the scores for each system on the samples obtained so far.
        Neither Stratum nor System objects are not updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum to score.

        Returns:
            list of float: A list of scores, one score per system
        """
        # If no trials in stratum, return nan
        if stratum.num_trials == 0:
            return [np.nan for system in stratum.system_list]

        score_df = stratum.get_combined_systems_score_df()
        # If no trials in key, return nan
        if score_df.shape[0] == 0:
            return [np.nan for system in stratum.system_list]
        # score_sample_adj = self.prior_samples_correction
        samples_list = self.estimate_samples_all_systems(stratum)
        pop_list = self.estimate_pop_all_systems(stratum)
        score_list = [0 for i in range(0, len(stratum.system_list))]
        for ind in range(0, len(stratum.system_list)):
            sys_id = stratum.system_list[ind].system_id
            if samples_list[ind] == 0:
                score_list[ind] = np.nan
            else:
                tp_df = score_df.loc[score_df[str(sys_id) + "_dec"] == self.high_key_value]
                tp = sum(tp_df[str(sys_id)+"_dec"] == tp_df["key"])
                num_samples = samples_list[ind]
                sys_pop = pop_list[ind]
                fpc = 0
                if sys_pop > 1:
                    fpc = 1 - ((num_samples - 1) / (sys_pop - 1))
                score_list[ind] = (tp + self.prior_samples_correction * fpc) / \
                    (samples_list[ind] + 2 * self.prior_samples_correction * fpc)
        return score_list

    def estimate_score_variance_all_systems(self, stratum):
        """
        Gives the score_variances for each system on the samples obtained so far.
        Neither Stratum nor System objects are not updated during this computation. The score
        variance is also enabled at the stratum level to allow for metric corrections such as
        priors for a bayesian confidence interval.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum of which to estimate
                the score variance.

        Returns:
            list of float: A list of score variances, with one score variance per system
            ordered by the order of systems in the system id list.

        """
        # If no trials in stratum, return nan
        if stratum.num_trials == 0:
            return [np.nan for system in stratum.system_list]

        score_df = stratum.get_combined_systems_score_df()
        # If no trials in key, return nan
        if score_df.shape[0] == 0:
            return [np.nan for system in stratum.system_list]

        samples_list = self.estimate_samples_all_systems(stratum)

        var_list = [0 for i in range(0, len(stratum.system_list))]
        for ind in range(0, len(stratum.system_list)):
            if samples_list[ind] == 0:
                var_list[ind] = np.nan
            else:
                sys_id = stratum.system_list[ind].system_id
                tp_df = score_df.loc[score_df[str(sys_id) + "_dec"] == self.high_key_value]
                tp = sum(tp_df[str(sys_id) + "_dec"] == tp_df["key"])
                fp = samples_list[ind] - tp
                alpha = tp + self.prior_samples_correction
                beta = fp + self.prior_samples_correction
                var_list[ind] = (alpha * beta) / (pow((alpha + beta), 2) * (alpha+beta+1))
        return var_list

    def get_actual_score(self, system_df, key_df):
        """
        Using a full system data frame and a key data frame, compute the score. This method is
        used when by the OracleScript that has a full key df. This computes the system Recall
        on the system. This assumes that the
        :func:`aegis.acteval.metrics.Metric.convert_thresholds_to_decision()` method has been
        called to produce the system_df

        Args:
            system_df (:obj:`pandas.core.frame.DataFrame`): The data frame of a single system.
                For metrics that have thresholds,
                it is assumed that this system has 'score' or 'decision' columns as needed.
            key_df (:obj:`pandas.core.frame.DataFrame`): The answer key data frame.

        Returns:
            float: A score for that system according to that key.

        """
        score_df = pd.merge(key_df, system_df, on="trial_id", how="left")
        system_recall = skm.recall_score(score_df["key"], score_df["decision"],
                                         pos_label=self.high_key_value)
        return system_recall


class BinaryAccuracyMetric(Metric):
    """
    Instantiation of a Metric corresponding to the simplest metric "accuracy", or percentage
    correct. Only for binary classification but it handles thresholds and arbitrary key values.

    With a threshold, it considers all of the values greater than the threshold to be in the
    high_key_value class with the remainder of examples to be in the low_key_value_class
    """
    low_key_value = 0
    high_key_value = 1

    def __init__(self, key_values=None, score_samples_correction=1,
                 score_variance_samples_correction=9):
        """
        Constructor.

        Args:
            key_values (list, optional): The two key values to use, specified as [low, high].
                None by default, which is replaced with [0, 1].
            score_samples_correction (float, optional): the adjustment on the score to correct the
                point estimate
                for proportions near 0 or 1. In computations, this number is divided by the
                square root of the number of samples. Defaults to 1.
            score_variance_samples_correction (float, optional): the adjustment on the score to
                correct
                the score variance estimate.
                for proportions near 0 or 1. In computations, this number is divided by the
                square root of the number of samples. This correction is
                added to the score_samples_correction to give additional width to the score
                variance. Defaults to 9.
        """
        if key_values is None:
            key_values = [0, 1]
        self.low_key_value = key_values[0]
        self.high_key_value = key_values[1]
        self.prior_samples_correction = 2
        self.score_samples_correction = score_samples_correction
        self.score_variance_samples_correction = score_variance_samples_correction
        super().__init__()

    def get_metric_name(self):
        """
        Returns the name of the metric as a string.

        Returns:
            str: the metric name

        """
        return "BinaryAccuracy"

    def find_needed_initial_samples(self, stratum, initial_samples_per_bin, rng):
        """
        In order to cover the case where we don't sample from each stratum necessary edge cases
        (Such as when the metric is a combination of two components) and to ensure samples
        from each stratum, this method will find such examples and will sample uniformly at random

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum object to find
                needed samples
            initial_samples_per_bin (int): The number of initial samples per bin
            rng (:obj:`numpy.random.RandomState`): random state object with stored random state

        Returns:
            list of object: A list of trial_id values to sample

        """
        if stratum.num_trials == 0:
            # No samples needed from an empty stratum
            return []

        combined_df = stratum.get_combined_systems_df()
        score_df = combined_df.loc[pd.notna(combined_df["key"]), :]
        non_score_df = combined_df.loc[pd.isna(combined_df["key"]), :]
        num_needed_samples = initial_samples_per_bin
        remaining_samples = num_needed_samples - score_df.shape[0]
        if remaining_samples <= 0:
            return []

        if non_score_df.shape[0] < remaining_samples:
            remaining_samples = non_score_df.shape[0]

        new_sample = non_score_df.sample(n=remaining_samples, replace=False, random_state=rng)
        return sorted(list(new_sample["trial_id"]))

    def convert_thresholds_to_decisions(self, system):
        """
        When computing Accuracy, we need to use the threshold specified to convert. This Accuracy
        assumes a binary classification problem. This converts to the high key value if the score
        is strictly higher than the threshold.

        The decision is added to the System data frame in the column "decision". The score
        column is preserved.

        Args:
            system (:obj:`aegis.acteval.system.System`): the system containing the scores

        Returns:
            aegis.acteval.system.System: The modified System object with a modified system_df with
            decisions in the "decision" column as well as preserving the score values.

        """

        # Important that we copy by reference
        sys_id = system.system_id
        sys_df = system.system_df
        # For now, use the first threshold value regardless of name
        threshold_df = system.threshold_df
        sys_threshold_val = threshold_df.loc[
            threshold_df["system_id"] == sys_id, "value"
        ].iloc[0]
        sys_df["decision"] = sys_df["score"] > sys_threshold_val
        sys_df["decision"] = sys_df["decision"].map({False: self.low_key_value,
                                                     True: self.high_key_value})
        return system

    def estimate_samples_all_systems(self, stratum):
        """
        Gets the number of sampled trials relevant to the metric for each system. For Binary
        Classification Accuracy, each trial sampled counts, so this returns the number of sampled
        trials in the stratum.
        Neither Stratum nor System objects are not updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum to determine how
                many samples count towards the population

        Returns:
            list of int: A list of samples counted for the systems, in the order of the
            systems in the stratum object.

        """
        if stratum.num_trials == 0:
            return [0 for i in range(0, len(stratum.system_list))]

        score_df = stratum.get_combined_systems_score_df()
        sys_sample_list = [
            score_df.shape[0] for i in range(0, len(stratum.system_list))
        ]
        return sys_sample_list

    def estimate_pop_all_systems(self, stratum):
        """
        Gives the total number of trials to be considered for the metric computation. For
        Binary classification of accuracy, it is simply the total number of trials in that stratum.


        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum of which to determine the
                population.

        Returns:
            list of int: A list of population sizes, with the ith element being the population
            size for system i.

        """
        stratum.population_size = stratum.num_trials
        sys_pop_list = [
            stratum.population_size for i in range(0, len(stratum.system_list))
        ]
        return sys_pop_list

    def estimate_pop_frac_variance_all_systems(self, stratum):
        """
        Estimate the uncertainty (variance) of our estimate of the population fraction. Returns
        0 when the population is known. For Accuracy, the population is known so this method
        Returns 0.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum of which to estimate
                the population fraction variance.

        Returns:
            list of float: A list of population fraction variances, with one estimate
            per system,
            ordered by the order of systems in the system id list.

        """
        sys_pop_var_list = [0 for i in range(0, len(stratum.system_list))]
        return sys_pop_var_list

    def get_trials_to_sample_from(self, stratum):
        """
        Given a stratum, determine the subset of trials that the stratum should draw from.
        If all samples are relevant, return all of the trial ids. For accuracy, all trials are
        relevant so all trials are returned.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): the stratum to identify which samples
            are relevant for sampling

        Returns:
            list of object: A list of trial ids to sample from

        """
        unsampled_trials = stratum.stratum_key_df.loc[pd.isna(stratum.stratum_key_df["key"]),
                                                      "trial_id"]
        return unsampled_trials

    def estimate_score_all_systems(self, stratum):
        """
        Gives the scores for each system on the samples obtained so far. The score here
        uses the accuracy_score() method from sklearn to compute the score as if it were
        a binomial value and we are computing the proportion of matches.
        Neither Stratum nor System objects are updated during this computation.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum to score.

        Returns:
            list of float: A list of scores, one score per system

        """
        # If no trials in stratum, return nan
        if stratum.num_trials == 0:
            return [np.nan for system in stratum.system_list]

        score_df = stratum.get_combined_systems_score_df()
        # If no trials in key, return nan
        if score_df.shape[0] == 0:
            return [np.nan for system in stratum.system_list]
        score_sample_adj = self.prior_samples_correction
        samples_list = self.estimate_samples_all_systems(stratum)
        pop_list = self.estimate_pop_all_systems(stratum)
        score_list = [0 for i in range(0, len(stratum.system_list))]
        for ind in range(0, len(stratum.system_list)):
            sys_id = stratum.system_list[ind].system_id
            acc_p = skm.accuracy_score(
                score_df["key"], score_df[str(sys_id) + "_dec"]
            )
            num_samples = samples_list[ind]
            sys_pop = pop_list[ind]
            if num_samples > 0:
                fpc = 0
                if sys_pop > 1:
                    fpc = 1 - ((num_samples - 1)/(sys_pop - 1))
                score_sample_adj = fpc*(self.score_samples_correction / np.sqrt(num_samples))
            corr_acc_p = (acc_p * num_samples + score_sample_adj) \
                / (num_samples + 2 * score_sample_adj)
            score_list[ind] = corr_acc_p
        return score_list

    def estimate_score_variance_all_systems(self, stratum):
        """
        Gives the score_variances for each system on the samples obtained so far.
        Neither Stratum nor System objects are not updated during this computation. The score
        variance is also enabled at the stratum level to allow for metric corrections such as
        priors for a bayesian confidence interval.

        Args:
            stratum (:obj:`aegis.acteval.stratum.Stratum`): The stratum of which to estimate
                the score variance.

        Returns:
            list of float: A list of score variances, with one score variance per system
            ordered by the order of systems in the system id list.

        """
        # If no trials in stratum, 0
        samples_list = self.estimate_samples_all_systems(stratum)
        pop_list = self.estimate_pop_all_systems(stratum)
        if stratum.num_trials == 0:
            return [0 for system in stratum.system_list]
        score_list = self.estimate_score_all_systems(stratum)
        score_var_list = [0 for i in range(0, len(stratum.system_list))]
        score_var_sample_adj = self.prior_samples_correction
        for ind in range(0, len(stratum.system_list)):
            acc_p = score_list[ind]
            if np.isnan(acc_p):
                score_var_list[ind] = 0
                continue
            num_samples = samples_list[ind]
            sys_pop = pop_list[ind]
            if num_samples > 0:
                fpc = 0
                if sys_pop > 1:
                    fpc = 1 - ((num_samples - 1) / (sys_pop - 1))
                score_sample_adj = fpc*(self.score_samples_correction / np.sqrt(num_samples))
                score_var_sample_adj = fpc * \
                    (self.score_variance_samples_correction/np.sqrt(num_samples))
            corr_acc_p = (acc_p * (num_samples + 2*score_sample_adj) + score_var_sample_adj) \
                / (num_samples + 2 * score_sample_adj + 2 * score_var_sample_adj)
            # Use binomial variance without a finite correction error.
            # In this case, the "n" is divided out by the number of trials
            try:
                score_var_list[ind] = corr_acc_p * (1 - corr_acc_p)
                score_var_list[ind] = score_var_list[ind] /\
                    (num_samples + 2*score_sample_adj + 2*score_var_sample_adj)
            except ZeroDivisionError:
                score_var_list[ind] = 0

        return score_var_list

    def get_actual_score(self, system_df, key_df):
        """
        Using a full system data frame and a key data frame, compute the score. This method is
        used when by the OracleScript that has a full key df. This computes the system Accuracy
        on the system. This assumes that the
        :meth:`aegis.acteval.metrics.Metric` method has been
        called to produce the system_df

        Args:
            system_df (:obj:`pandas.core.frame.DataFrame`): The data frame of a single system.
                For metrics that have thresholds,
                it is assumed that this system has 'score' or 'decision' columns as needed.
            key_df (:obj:`pandas.core.frame.DataFrame`): The answer key data frame.

        Returns:
            float: A score for that system according to that key.

        """
        score_df = pd.merge(key_df, system_df, on="trial_id", how="left")
        system_accuracy = skm.accuracy_score(score_df["key"], score_df["decision"])
        return system_accuracy
