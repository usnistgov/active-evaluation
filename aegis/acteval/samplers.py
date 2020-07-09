import numpy as np
import pandas as pd
import abc


class TrialSampler(abc.ABC):
    """ Abstract class to support different sampling methods."""

    name = ""

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__()
        self.strata = strata
        self.num_success_rounds_required = num_success_rounds_required

    @abc.abstractmethod
    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Method that provides the strategy of drawing samples. Will vary when needed for
        comparative experiments.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        pass

    def sample_next_round(self, num_previous_successes):
        """
        Method to determine if we should sample another round.

        Args:
            num_previous_successes (int): The number of previous consecutive successful rounds

        Returns:
            bool: True if another round should be sampled, and False otherwise.

        """
        return num_previous_successes < self.num_success_rounds_required

    def meets_confidence_delta(self, conf_value, delta):
        """
        Submethod useful for testing if we meet the confidence criteria from the Sampler
        object rather than the Strata object.

        Args:
            conf_value (float): The previously computed confidence value
            delta (float): The delta of uncertainty to be within.

        Returns:
            bool: True if the confidence range is within the specified delta
            False, otherwise

        """
        return conf_value <= delta

    def meets_confidence_criteria(self, strata, delta, alpha, metric):
        """
        Takes a strata, with its strategy to aggregate the confidence intervals of the different
        systems in the strata, and a metric to compute confidence intervals, and returns if with
        probability >= 1-alpha the confidence interval width is bounded by delta.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object
            delta (float): The delta to be within
            alpha (float): The alpha parameter of the confidence interval (or the probability
                level of the confidence interval)
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric according to which to
                compute the confidence criteria

        Returns:
            bool: meets_criteria a boolean that determines if the sample meets the
            confidence criteria within the specified parameters.

        """

        # Compute confidence values here.
        strata.get_confidence_intervals_all_systems(metric, alpha)
        delta_pref_var = strata.aggregate_system_confidence_values()
        return self.meets_confidence_delta(delta_pref_var, delta)


class UniformFixedTrialSampler(TrialSampler):
    """
    Draws samples from the strata uniformly, drawing the same number of samples from each
    stratum. This does not use a multinomial distribution but a deterministic
    """

    name = "UniformFixedTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Samples uniformly from each strata, meaning that each strata will get
        num_samples/self.strata.num_strata samples.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        num_strata = self.strata.num_strata

        # Allocate samples uniformly over all of the strata
        pvals = [1.0 / num_strata] * num_strata
        nk = [np.floor(pval * num_samples).astype(int) for pval in pvals]
        # Additional check added to make sure total number of samples are met
        if sum(nk) < num_samples:
            missing_samples = (num_samples-sum(nk)) * -1
            full_nums = [pval * num_samples for pval in pvals]
            frac_nums = [my_num - np.floor(my_num) for my_num in full_nums]
            sample_index = np.argsort(frac_nums)[missing_samples:]
            for i in sample_index:
                nk[i] += 1
        # Now that the number of samples per strata are computed, draw the samples
        new_samples = []
        # We use a for loop instead of a list comprehension to ensure that the rng is updated
        # after each sampling
        for stratum, i in zip(self.strata.strata, nk):
            samples = stratum.draw_samples(i, metric, rng)
            # This does not check for duplicates
            new_samples.extend(samples)
        return new_samples


class UniformTrialSampler(TrialSampler):
    """
    Draws samples from the strata uniformly, drawing from a uniform multinomial. This uniform
    sampler draws a deterministic number of samples rather than using a multinomial distribution
    with the uniform probabilities.
    """
    name = "UniformTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Samples uniformly from each strata, meaning that each strata will get
        num_samples/self.strata.num_strata samples.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        num_strata = self.strata.num_strata

        # Allocate samples uniformly over all of the strata
        pvals = [1.0 / num_strata] * num_strata
        nk = list(rng.multinomial(num_samples, pvals, size=1))[0]
        # Now that the number of samples per strata are computed, draw the samples
        new_samples = []
        # We use a for loop instead of a list comprehension to ensure that the rng is updated
        # after each sampling
        for stratum, i in zip(self.strata.strata, nk):
            samples = stratum.draw_samples(i, metric, rng)
            # This does not check for duplicates
            new_samples.extend(samples)
        return new_samples


class ProportionalFixedTrialSampler(TrialSampler):
    """
    Draws samples from the strata, sampling proportionally to the population size of each stratum.
    This uses the population estimates or the populations according to the metrics, which for
    some metrics may not be the number of trials. This proportional sampler draws a deterministic
    number of samples rather than using a multinomial distribution with the proportional
    probabilities.
    """
    name = "ProportionalFixedTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Sample proportional to the stratum population size. Population will be estimated for
        metrics like precision where the population is not merely the number of trials in
        the stratum.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        num_strata = self.strata.num_strata

        # Use populations to sample proportional to strata populations
        normalizer = np.nansum(
            [
                i.estimate_pop_upper(metric, self.strata.aggregate_system_stats, alpha)
                for i in self.strata.strata
            ]
        )
        if normalizer == 0.0:
            pvals = [1.0 / num_strata] * num_strata
        else:
            try:
                pvals = [
                    i.estimate_pop_upper(metric, self.strata.aggregate_system_stats, alpha)
                    / normalizer
                    for i in self.strata.strata
                ]
                # All nan's give a RuntimeWarning but not an error, so check for an Nan
                if True in np.isnan(pvals):
                    pvals = [1.0 / num_strata] * num_strata
            except ZeroDivisionError:
                pvals = [1.0 / num_strata] * num_strata

        nk = [np.floor(pval * num_samples).astype(int) for pval in pvals]
        # Additional check added to make sure total number of samples are met
        if sum(nk) < num_samples:
            missing_samples = (num_samples-sum(nk)) * -1
            full_nums = [pval * num_samples for pval in pvals]
            frac_nums = [my_num - np.floor(my_num) for my_num in full_nums]
            sample_index = np.argsort(frac_nums)[missing_samples:]
            for i in sample_index:
                nk[i] += 1
        # Now that the number of samples per strata are computed, draw the samples
        new_samples = []
        # We use a for loop instead of a list comprehension to ensure that the rng is updated
        # after each sampling
        for stratum, i in zip(self.strata.strata, nk):
            samples = stratum.draw_samples(i, metric, rng)
            # This does not check for duplicates
            new_samples.extend(samples)
        return new_samples


class ProportionalTrialSampler(TrialSampler):
    """
    Draws samples from the strata, sampling proportionally to the population size of each stratum.
    This uses the population estimates or the populations according to the metrics, which for
    some metrics may not be the number of trials.
    """
    name = "ProportionalTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Sample proportional to the stratum population size. Population will be estimated for
        metrics like precision where the population is not merely the number of trials in
        the stratum.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        num_strata = self.strata.num_strata

        # Use populations to sample proportional to strata populations
        normalizer = np.nansum(
            [
                i.estimate_pop_upper(metric, self.strata.aggregate_system_stats, alpha)
                for i in self.strata.strata
            ]
        )
        if normalizer == 0.0:
            pvals = [1.0 / num_strata] * num_strata
        else:
            try:
                pvals = [
                    i.estimate_pop_upper(metric, self.strata.aggregate_system_stats, alpha)
                    / normalizer
                    for i in self.strata.strata
                ]
                # All nan's give a RuntimeWarning but not an error, so check for an Nan
                if True in np.isnan(pvals):
                    pvals = [1.0 / num_strata] * num_strata
            except ZeroDivisionError:
                pvals = [1.0 / num_strata] * num_strata

        nk = list(rng.multinomial(num_samples, pvals, size=1))[0]
        # Now that the number of samples per strata are computed, draw the samples
        new_samples = []
        # We use a for loop instead of a list comprehension to ensure that the rng is updated
        # after each sampling
        for stratum, i in zip(self.strata.strata, nk):
            samples = stratum.draw_samples(i, metric, rng)
            # This does not check for duplicates
            new_samples.extend(samples)
        return new_samples


class AdaptiveFixedTrialSampler(TrialSampler):
    """
    Draws samples from the strata, corresponding to the opt sampler that samples
    proportionally according to the variance of each stratum. Draws proportional exactly
    instead of using a multinomial.
    """
    name = "AdaptiveFixedTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Sample adaptively according to an "optimal" sampling scheme, which is sampling
        proportional to the score_variances of each stratum. This enables taking more samples
        for a stratum with a higher variance.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        num_strata = self.strata.num_strata
        normalizer = np.nansum(
            [
                i.estimate_pop_upper(metric, self.strata.aggregate_system_stats, alpha)
                * np.sqrt(
                    i.estimate_score_variance_upper(
                        metric, self.strata.aggregate_system_stats
                    )
                )
                for i in self.strata.strata
            ]
        )
        # Handle case where normalizer is 0
        if normalizer == 0.0:
            pvals = [1.0 / num_strata] * num_strata
        else:
            try:
                pvals = [
                    i.estimate_pop_upper(metric, self.strata.aggregate_system_stats, alpha)
                    * np.sqrt(
                        i.estimate_score_variance_upper(
                            metric, self.strata.aggregate_system_stats
                        )
                    )
                    / normalizer
                    for i in self.strata.strata
                ]
                # All nan's give a RuntimeWarning but not an error, so check for an Nan
                if True in np.isnan(pvals):
                    pvals = [1.0 / num_strata] * num_strata
            except ZeroDivisionError:
                pvals = [1.0 / num_strata] * num_strata

        nk = [np.floor(pval * num_samples).astype(int) for pval in pvals]
        # Additional check added to make sure total number of samples are met
        if sum(nk) < num_samples:
            missing_samples = (num_samples - sum(nk)) * -1
            full_nums = [pval * num_samples for pval in pvals]
            frac_nums = [my_num - np.floor(my_num) for my_num in full_nums]
            sample_index = np.argsort(frac_nums)[missing_samples:]
            for i in sample_index:
                nk[i] += 1
        # Now that the number of samples per strata are computed, draw the samples
        new_samples = []
        # We use a for loop instead of a list comprehension to ensure that the rng is updated
        # after each sampling
        for stratum, i in zip(self.strata.strata, nk):
            samples = stratum.draw_samples(i, metric, rng)
            # This does not check for duplicates
            new_samples.extend(samples)
        return new_samples


class AdaptiveTrialSampler(TrialSampler):
    """
    Draws samples from the strata, corresponding to the opt sampler that samples
    proportionally according to the variance of each stratum.
    """
    name = "AdaptiveTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Sample adaptively according to an "optimal" sampling scheme, which is sampling
        proportional to the score_variances of each stratum. This enables taking more samples
        for a stratum with a higher variance.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        num_strata = self.strata.num_strata
        normalizer = np.nansum(
            [
                i.estimate_pop_upper(metric, self.strata.aggregate_system_stats, alpha)
                * np.sqrt(
                    i.estimate_score_variance_upper(
                        metric, self.strata.aggregate_system_stats
                    )
                )
                for i in self.strata.strata
            ]
        )
        # Handle case where normalizer is 0
        if normalizer == 0.0:
            pvals = [1.0 / num_strata] * num_strata
        else:
            try:
                pvals = [
                    i.estimate_pop_upper(metric, self.strata.aggregate_system_stats, alpha)
                    * np.sqrt(
                        i.estimate_score_variance_upper(
                            metric, self.strata.aggregate_system_stats
                        )
                    )
                    / normalizer
                    for i in self.strata.strata
                ]
                # All nan's give a RuntimeWarning but not an error, so check for an Nan
                if True in np.isnan(pvals):
                    pvals = [1.0 / num_strata] * num_strata
            except ZeroDivisionError:
                pvals = [1.0 / num_strata] * num_strata

        nk = list(rng.multinomial(num_samples, pvals, size=1))[0]
        # Now that the number of samples per strata are computed, draw the samples
        new_samples = []
        # We use a for loop instead of a list comprehension to ensure that the rng is updated
        # after each sampling
        for stratum, i in zip(self.strata.strata, nk):
            samples = stratum.draw_samples(i, metric, rng)
            # This does not check for duplicates
            new_samples.extend(samples)
        return new_samples


class RandomFixedTrialSampler(TrialSampler):
    """
    Draws samples randomly regardless of stratification. Calls
    the metric get_trials_to_sample_from() in order to only sample relevant trials.
    """

    name = "RandomFixedTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Samples randomly regardless of stratification. However, it does call the metric
        `get_trials_to_sample_from` for each stratum so that all sampled trials are relevant.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        unsampled_trials = metric.get_trials_to_sample_from(self.strata.strata[0]).copy(deep=True)
        for stratum in self.strata.strata[1:]:
            unsampled_temp = metric.get_trials_to_sample_from(stratum)
            unsampled_trials = unsampled_trials.append(unsampled_temp)

        if num_samples > unsampled_trials.shape[0]:
            if unsampled_trials.shape[0] == 0:
                return []
            else:
                num_samples = unsampled_trials.shape[0]

        samples = unsampled_trials.sample(n=num_samples, replace=False, random_state=rng)
        return samples


class RandomTrialSampler(TrialSampler):
    """
    Draws samples randomly regardless of stratification. RandomTrialSampler calls
    the metric get_trials_to_sample_from() in order to only sample relevant trials.
    """
    name = "RandomTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Samples randomly regardless of stratification. However, it does call the metric
        `get_trials_to_sample_from` for each stratum so that all sampled trials are relevant.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        # Get all of the samples
        unsampled_trials = metric.get_trials_to_sample_from(self.strata.strata[0]).copy(deep=True)
        for stratum in self.strata.strata[1:]:
            unsampled_temp = metric.get_trials_to_sample_from(stratum)
            unsampled_trials = unsampled_trials.append(unsampled_temp)

        if num_samples > unsampled_trials.shape[0]:
            if unsampled_trials.shape[0] == 0:
                return []
            else:
                num_samples = unsampled_trials.shape[0]

        samples = unsampled_trials.sample(n=num_samples, replace=False, random_state=rng)
        return samples


class TrueRandomFixedTrialSampler(TrialSampler):
    """
    Draws samples randomly regardless of stratification. Samples from
    all trials without consulting the metric. This means that in many iterations non-relevant
    trials may be sampled for metrics such as precision where not all trials are relevant.
    """

    name = "TrueRandomFixedTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Samples from all stratum from all possible unsampled trials. This includes trials that
        the metric may deem as not relevant.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        stratum = self.strata.strata[0]
        unsampled_trials = stratum.stratum_key_df.loc[pd.isna(stratum.stratum_key_df["key"]),
                                                      "trial_id"].copy(deep=True)
        for stratum in self.strata.strata[1:]:
            unsampled_temp = stratum.stratum_key_df.loc[pd.isna(stratum.stratum_key_df["key"]),
                                                        "trial_id"]
            unsampled_trials = unsampled_trials.append(unsampled_temp)

        if num_samples > unsampled_trials.shape[0]:
            if unsampled_trials.shape[0] == 0:
                return []
            else:
                num_samples = unsampled_trials.shape[0]

        samples = unsampled_trials.sample(n=num_samples, replace=False, random_state=rng)
        return samples


class TrueRandomTrialSampler(TrialSampler):
    """
    Draws samples randomly regardless of stratification. Samples from all trials,
    regardless of whether they are relevant or not.
    """
    name = "TrueRandomTrialSampler"

    def __init__(self, strata, num_success_rounds_required):
        """
        Sampler constructor.

        Args:
            strata (:obj:`aegis.acteval.strata.Strata`): The strata object that the sampler is
                sampling from
            num_success_rounds_required (int): the number of consecutive rounds needed to sample
                before stopping sampling
        """
        super().__init__(strata, num_success_rounds_required)

    def draw_samples(self, num_samples, metric, rng=None, alpha=0.05):
        """
        Samples randomly regardless of stratification. Draws from all samples, including
        some that may not be relevant.

        Args:
            num_samples (int): The number of samples to draw at each step
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object used for scoring
            rng (:obj:`numpy.random.RandomState`, optional): The Random_number generator provided
                from the controller. If set to None, it is
                not provided and will be unique to this sampler. None is provided by default
            alpha (float, optional): The desired confidence level. Used for sampling when
                population estimates have variances

        Returns:
            list of object: A list of trial_id values to sample.

        """
        # Get all of the samples
        stratum = self.strata.strata[0]
        unsampled_trials = stratum.stratum_key_df.loc[pd.isna(stratum.stratum_key_df["key"]),
                                                      "trial_id"].copy(deep=True)
        for stratum in self.strata.strata[1:]:
            unsampled_temp = stratum.stratum_key_df.loc[pd.isna(stratum.stratum_key_df["key"]),
                                                        "trial_id"]
            unsampled_trials = unsampled_trials.append(unsampled_temp)

        if num_samples > unsampled_trials.shape[0]:
            if unsampled_trials.shape[0] == 0:
                return []
            else:
                num_samples = unsampled_trials.shape[0]

        samples = unsampled_trials.sample(n=num_samples, replace=False, random_state=rng)
        return samples
