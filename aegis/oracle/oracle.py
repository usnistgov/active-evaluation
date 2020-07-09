import pandas as pd
import abc
import numpy as np
import os
import joblib
import datetime
import itertools
import logging
import glob

import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.acteval.controller
import aegis.acteval.experiment


def compute_var(row, n):
    """
    Method to compute var, using Using Var(x) = E(X^2) - E(X)^2.
    Extracts columns 'total_score_diff_sq' and `avg_score_diff` from data frame row.

    Because of floating point computations, variance may be slightly negative (near 0)

    Args:
        row (:obj:`pandas.core.series.Series`): row of data frame
        n (int): the number of examples

    Returns:
        float: A computed variance (square of the standard deviation).

    """
    return (row['total_score_diff_sq'] / float(n)) - np.square(row['avg_score_diff'])


class Oracle(abc.ABC):

    def __init__(self):
        """ Constructor. """
        logger = logging.getLogger('paper_experiments_logger')
        logger.info('Initiating oracle class')
        super().__init__()

    @abc.abstractmethod
    def get_annotations(self, trial_samples):
        """
        Returns a (trial_id, key) pandas data frame with just the suggested trials

        Args:
            trial_samples (list of object): A list of trial_id values to get the annotations for

        Returns:
            pandas.core.frame.DataFrame: A two-column data frame with columns "trial_id" and "key".
            Resets the index
            so that the index does not inadvertently contain information.

        """


class OracleScript(Oracle):
    """
    Oracle Script that assumes we have the ground truth for all trials stored separately.
    This Oracle Script is useful for automating experiments when results are known.
    Because OracleScript objects have a full key, we implement methods useful for experimental
    simulations that leverage our knowledge of the full key on the data set.
    """

    # Do we want to call the data processor for the oracle class? Right now, we do not
    # oracle_data_processor = acteval.data_processor.DataProcessor()
    key_df = pd.DataFrame()
    """
    pandas.DataFrame: The read in data frame of the answer key.

    Two columns: "trial_id" and "key".
    """

    def __init__(self, key_file):
        """
        Constructor.
        Args:
            key_file (string): The filepath to the key data frame. The data frame should have two
                columns: "trial_id" and "key", with the key corresponding to that trial id.
        """
        self.key_df = pd.read_csv(key_file)
        super().__init__()

    def get_key_df_copy(self):
        """
        Returns a deep copy of the data frame.

        Returns:
            pandas.core.frame.DataFrame: A copy of the key data frame.

        """
        return self.key_df.copy(deep=True)

    def get_annotations(self, trial_samples):
        """
        Returns a (trial_id, key) pandas data frame with just the suggested trials.
        This implementation takes the relevant rows from the key file.

        Args:
            trial_samples (list of object): A list of trial_id values to get the annotations for

        Returns:
            pandas.core.frame.DataFrame: A two-column data frame with columns "trial_id" and "key".
            Resets the index so that the index does not inadvertently contain information.

        """
        annotation_df = self.key_df.loc[
            self.key_df["trial_id"].isin(trial_samples), :
        ].reset_index(drop=True)
        return annotation_df

    def get_actual_score_all_systems(self, system_fpaths, threshold_fpaths, metric):
        """
        Uses the full key file to get the actual score for each system with the specified metric
        object.

        Args:
            system_fpaths (list of str): The list of filepaths to the systems
            threshold_fpaths (list of str): The list of filepaths to the thresholds
            metric (:obj:`aegis.acteval.metrics.Metric`): The metric object that will
                score the systems

        Returns:
            list of float: A list of scores, one per system in the order of the files in
            system_fpaths


        """
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        [metric.convert_thresholds_to_decisions(system) for system in system_list]

        system_df_list = [system.system_df for system in system_list]
        system_accuracy_list = [metric.get_actual_score(system_df, self.key_df)
                                for system_df in system_df_list]
        return system_accuracy_list

    def is_within_ci_all_systems(self, system_list, actual_score_list):
        """
        Given the system objects and their actual system scores, provide a list of booleans
        that tells us if each system's estimate confidence interval contains the actual score.

        Args:
            system_list (list of :obj:`aegis.acteval.system.System`): The list of system objects
            actual_score_list (list of float): The actual score of the systems

        Returns:
            list of bool: A list of boolean values that each entry is True if the true score
            is contained in the
            confidence interval for that system, and False otherwise.

        """
        return [(system.score - system.confidence_value) <= score <=
                (system.score + system.confidence_value)
                for (system, score) in zip(system_list, actual_score_list)]

    @staticmethod
    def system_within_ci(system, actual_score):
        """
        Given a single system object and its actual core, provides a boolean
        that tells us if that system's estimate confidence interval contains the actual score.

        Args:
            system (:obj:`aegis.acteval.system.System`): The system object
            actual_score (float): The actual score of the system

        Returns:
            bool: A boolean values that is True if the true score is contained in the
            confidence interval for that system, and False otherwise.


        """
        return (system.score - system.confidence_value) <= actual_score <= \
               (system.score + system.confidence_value)

    def all_within_ci(self, system_list, actual_score_list):
        """
        Given a list system objects and their actual system scores, provides a list of booleans
        that tells us if each system's estimate confidence interval contains the actual score.

        Args:
            system_list (list of :obj:`aegis.acteval.system.System`): The list of system objects
            actual_score_list (list of float): The actual score of the systems

        Returns:
            list of bool: A list of boolean values where each entry is True if the true score is
            contained in the
            confidence interval for that system, and False otherwise.

        """
        ci_list = [OracleScript.system_within_ci(system, score)
                   for (system, score) in zip(system_list, actual_score_list)]
        return all(ci_list)

    @staticmethod
    def system_within_delta(system, actual_score, delta):
        """
        Given a single system object and its actual core, provide a boolean
        that tells us if that system's estimate is within +- delta of the actual score

        Args:
            system (:obj:`aegis.acteval.system.System`): The system object
            actual_score (float): The actual score of the system
            delta (float): The specified error range

        Returns:
            bool: A boolean values that is True if the system's true score is
            contained in the
            interval of system estimate +- delta, and False otherwise.


        """
        return (system.score - delta) <= actual_score <= \
               (system.score + delta)

    def all_within_delta(self, system_list, actual_score_list, delta):
        """
        Given the system objects and their actual system scores, provide a list of booleans
        that tells us if each system's estimate +- delta contains the actual score.

        Args:
            system_list (list of :obj:`aegis.acteval.system.System`): The list of system objects
            actual_score_list (list of float): The actual score of the systems
            delta (float): the specified delta value

        Returns:
            list of bool: A list of boolean values where each entry is True if the true score is
            contained in the
            interval of system estimate is +- delta, and False otherwise.

        """
        ci_list = [OracleScript.system_within_delta(system, score, delta)
                   for (system, score) in zip(system_list, actual_score_list)]
        return all(ci_list)

    @staticmethod
    def get_specific_experiment_result_data(oracle_ref, input_dir, metric_object,
                                            stratification_type,
                                            num_strata, sampler_types=None, sampler_categories=None,
                                            system_ordering=None, bin_style_types=None,
                                            use_multinomials=None,
                                            num_success_rounds_required=3,
                                            num_step_samples=100, alpha=0.05, delta=0.01,
                                            num_iterations=1000, use_initial_df=True,
                                            request_initial_samples=True,
                                            initial_samples=10, parallelize=True,
                                            random_seed=None):
        """
            The component of "run_specific_experiment" that runs the data frames and returns
            the data frames.

            Args:
                oracle_ref (:obj:`aegis.oracle.oracle.Oracle`):
                    the Oracle object to call to get additional trials
                input_dir (str): The path to the input directory of the submission
                metric_object (:obj:`aegis.acteval.metrics.Metric`):
                    The reference to the metric object that specifies how the trials will be scored
                stratification_type (:obj:`aegis.acteval.strata.Strata`):
                    Strata class that gives the stratification strategy and type of strata
                num_strata (int): The number of strata to have.
                sampler_types (list of :obj:`abc.ABCMeta`, optional):
                    List of samplers you wish to use in experiment
                sampler_categories (list of str, optional): List of sampler categories you
                    want to use in experiment. Defaults to None, which resolves as [].
                system_ordering (list of str): The desired ordering of systems, if the order
                    is important. Order is specified by system id.
                    None, which resolves to [] By
                    default. One can also give an ordering of only some of the systems to run the
                    experiment on a subset of systems in the submission. [] Runs all systems in an
                    arbitrary ordering.
                bin_style_types (list of str, optional): List of bin style types you want to
                    use in experiment. Defaults to None, which resolves as [].
                use_multinomials (list of bool, optional): List of booleans for each sampler type
                    Defaults to None, which resolves as [].
                num_success_rounds_required (int, optional):
                    The number of rounds where the (1-\\alpha) confidence
                    interval's range is within +- $\\delta width. Defaults to 2
                num_step_samples (int, optional): the number of samples to ask for at each iteration
                    defaults to 100
                alpha (num, optional): The specified probability \\alpha. Defaults to 0.05.
                delta (num, optional): The specified interval range \\delta. Defaults to 0.01
                num_iterations (int, optional): The number of iterations to run each sampler for.
                    Defaults to 1000
                use_initial_df (bool, optional): A variable that tells is to use an initial
                    data frame if provided
                    for initial samples. Defaults to True.
                request_initial_samples (bool, optional):
                    A boolean that determines if the experiment (and hence) the
                    controller should request or supplement initial samples with initial samples
                    to provide for adequate stratum and metric coverage for initial estimates.
                    True by default.
                initial_samples (int, optional): The number of samples to request initially.
                    Defaults to 100.
                parallelize (bool, optional): A boolean determining whether to parallelize. True
                    parallelizes with joblib, False runs in serial. Defaults to True
                random_seed (int, optional): The desired random seed. Set to None if it is desired
                    for the experiment to not be seeded. Defaults to None.

            Returns:
                (pandas.core.frame.DataFrame, pandas.core.frame.Dataframe): The two data
                frames (summary_out_df, results_out_df) as a tuple so that they
                can be written to a file.

        """
        # Place default arguments here so that no default arguments are mutable
        if sampler_types is None:
            sampler_types = []

        if sampler_categories is None:
            sampler_categories = []

        if system_ordering is None:
            system_ordering = []

        if bin_style_types is None:
            bin_style_types = []

        if use_multinomials is None:
            use_multinomials = []

        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        num_systems = len(system_fpaths)
        dataset_name = os.path.basename(input_dir)
        sys_order_str = str(system_ordering)
        # Experimental parameters that are constant across all experiments
        my_metric = metric_object

        my_controller = aegis.acteval.controller.Controller()

        # Score the system against the key to get the accuracy
        system_accuracy_list = oracle_ref.get_actual_score_all_systems(system_fpaths,
                                                                       threshold_fpaths,
                                                                       my_metric)
        # Experimental parameters that vary
        sampler_type_list = sampler_types
        sampler_category = sampler_categories
        bin_style_types = bin_style_types
        uses_multinomial = use_multinomials

        sampler_df = pd.DataFrame(sampler_type_list, columns=["sampler_type"])
        sampler_df['sampler_category'] = sampler_category
        sampler_df['uses_multinomial'] = uses_multinomial
        bin_df = pd.DataFrame(bin_style_types, columns=["bin_style"])

        # For summary data frame
        summary_df = sampler_df.assign(foo=1).merge(bin_df.assign(foo=1)).drop('foo', 1)
        summary_df['total_samples'] = 0
        summary_df['total_relevant_samples'] = 0
        summary_df['num_within_ci'] = 0
        summary_df['num_within_delta'] = 0
        summary_df['total_score_diff'] = 0
        summary_df['total_score_diff_sq'] = 0
        summary_df['sampler_str'] = summary_df["sampler_type"].apply(str)
        summary_df['metric_name'] = metric_object.get_metric_name()
        summary_df['dataset'] = dataset_name
        summary_df['system_ordering'] = sys_order_str
        summary_df['stratification_type'] = stratification_type

        # Do not feed initial files if desired to not use
        if not use_initial_df:
            init_fpath = None

        def process_iteration(iter_num, sampler_type, bin_style, stratification_type,
                              num_strata,
                              submission_dirname, sys_ordering,
                              random_state=None):
            """
            Process an experiment. Most parameters for the experiment come from the experiment file
            in which this method is defined. This method, in addition to returning summarized output
            as a 4-tuple, writes the fuller experiment summary to a text file.

            Args:
                iter_num (int): the current iteration. Used by joblib and to identify file output
                sampler_type (:obj:`abc.ABCMeta`): the sampler type for the experiment.
                bin_style (str): the bin style for the stratifications.
                stratification_type (:obj:`abc.ABCMeta`): The type of stratification used
                num_strata: the number of strata to use
                submission_dirname (str): The name of the submission directory
                sys_ordering (str): The system ordering provided
                random_state (:obj:`numpy.random.RandomState`, optional): The random state based
                    on the seed. If None, then the randomization
                    is not seeded. Defaults to None.

            Returns:
                (int, abc.ABCMeta, str, pandas.core.series.Series):
                Rows as a list of tuples that can be easily converted to a rows in a
                pandas data frame, one row per tuple.
                (iter_num, sampler_type, bin_style, report_tuple) as an expanded tuple. Returns
                one row per system for ease of processing.

            """
            # Give a new random state
            if random_state is None:
                rng = None
            else:
                # np.random.seed(seed=random_state)
                rng = np.random.RandomState(random_state)
            # Handle UniformRandom case with this preamble block
            my_sampler = sampler_type

            sampler_name = sampler_type.name
            if my_sampler == aegis.acteval.samplers.RandomTrialSampler:
                num_strata = -1
            elif my_sampler == aegis.acteval.samplers.RandomFixedTrialSampler:
                num_strata = -1

            # run experiment and write the outputs
            my_experiment = aegis.acteval.experiment. \
                ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                                 num_success_rounds_required=num_success_rounds_required,
                                 num_strata=num_strata,
                                 metric_object=metric_object,
                                 stratification_type=stratification_type,
                                 sampler_type=my_sampler,
                                 bin_style=bin_style,
                                 request_initial_samples=request_initial_samples,
                                 initial_samples=initial_samples)
            exp_report = my_controller.run(init_fpath, trial_data_fpath,
                                           system_fpaths, threshold_fpaths,
                                           oracle_ref, my_experiment, rng)
            all_within_ci = oracle_ref.all_within_ci(exp_report.system_list,
                                                     system_accuracy_list)
            all_within_delta = oracle_ref.all_within_delta(exp_report.system_list,
                                                           system_accuracy_list, delta)
            ci_list = [(aegis.oracle.oracle.OracleScript.system_within_ci(system, score), score)
                       for (system, score) in zip(exp_report.system_list, system_accuracy_list)]
            delta_list = \
                [(aegis.oracle.oracle.OracleScript.system_within_delta(system, score, delta),)
                 for (system, score) in zip(exp_report.system_list, system_accuracy_list)]
            # Take exp_report and return a tuple of items to go directly as a row in our data_frame
            report_tuple = exp_report.get_experiment_report_tuple()
            metric_object.get_metric_name()
            sec_fac_tuple = (my_experiment.metric_object.get_metric_name(), submission_dirname,
                             sys_ordering)
            systems_tuple_list = exp_report.get_systems_tuples()
            iteration_tuple = (iter_num, sampler_type, sampler_name, bin_style, all_within_ci,
                               all_within_delta)
            iteration_tuple_list = [iteration_tuple + report_tuple + system_tuple +
                                    system_within_ci + system_within_delta + sec_fac_tuple
                                    for (system_tuple, system_within_ci, system_within_delta)
                                    in zip(systems_tuple_list, ci_list, delta_list)]
            return iteration_tuple_list

        if not (random_seed is None):
            np.random.seed(seed=random_seed)

        rn_state_vec = np.random.randint(np.iinfo(np.int32).max,
                                         size=num_iterations * len(sampler_type_list) *
                                         len(bin_style_types))
        all_iter_lists = [range(1, num_iterations + 1), sampler_type_list,
                          bin_style_types]
        iteration_cp = list(itertools.product(*all_iter_lists))
        iteration_prod = [(rn,) + iter_item for (rn, iter_item)
                          in zip(rn_state_vec, iteration_cp)]
        if parallelize:
            results = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(process_iteration)(iter_num=i, sampler_type=sampler_type,
                                                  bin_style=bin_style,
                                                  stratification_type=stratification_type,
                                                  num_strata=num_strata,
                                                  submission_dirname=dataset_name,
                                                  sys_ordering=sys_order_str,
                                                  random_state=random_state)
                for random_state, i, sampler_type, bin_style in iteration_prod)
        else:
            results = []
            for random_state, i, sampler_type, bin_style in iteration_prod:
                results.append(process_iteration(iter_num=i,
                                                 sampler_type=sampler_type,
                                                 bin_style=bin_style,
                                                 stratification_type=stratification_type,
                                                 num_strata=num_strata,
                                                 submission_dirname=dataset_name,
                                                 sys_ordering=sys_order_str,
                                                 random_state=random_state))

        # Get a nice data frame
        results_header = ["iter_num", "sampler_type", "sampler_name", "bin_style", "all_within_ci",
                          "all_within_delta",
                          "num_step_samples",
                          "num_success_rounds_required", "alpha", "delta", "num_strata",
                          "stratification_type", "exp_bin_style", "metric_object",
                          "exp_sampler_type",
                          "request_initial_samples", "initial_samples",
                          "total_sampled_trials", "num_rounds",
                          "num_init_trials", "num_requested_init_trials", "num_nonempty_strata",
                          "system_id", "score", "confidence_value", "score_variance",
                          "sampled_trials", "population", "system_within_ci",
                          "actual_system_score", "system_within_delta",
                          "metric_name", "dataset", "system_ordering"]
        # Using
        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
        # to flatten lists
        results_rows = []
        for sublist in results:
            for item in sublist:
                results_rows.append(item)
        results_df = pd.DataFrame(results_rows, columns=results_header)
        results_df['score_diff'] = results_df['score'] - results_df['actual_system_score']
        results_df['score_diff_sq'] = results_df['score_diff'].apply(np.square)

        for index, row in results_df.iterrows():
            # Check within delta not within the system's estimate of the CI for the summary
            sampler_str, bin_style, all_within_ci, all_within_delta, sampled_trials, score_diff,\
                total_sampled_trials = \
                str(row["exp_sampler_type"]), row["bin_style"], row["all_within_ci"], \
                row["all_within_delta"], row["sampled_trials"], row["score_diff"], \
                row["total_sampled_trials"]
            if all_within_delta:
                summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                               (summary_df['bin_style'] == bin_style), 'num_within_delta'] += 1
            if all_within_ci:
                summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                               (summary_df['bin_style'] == bin_style), 'num_within_ci'] += 1
            summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                           (summary_df['bin_style'] == bin_style), 'total_relevant_samples'] += \
                sampled_trials
            summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                           (summary_df['bin_style'] == bin_style), 'total_samples'] += \
                total_sampled_trials
            # For score differences take all of the systems
            # and then divide by num_iterations*num_systems
            summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                           (summary_df['bin_style'] == bin_style), 'total_score_diff'] += \
                score_diff
            summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                           (summary_df['bin_style'] == bin_style), 'total_score_diff_sq'] += \
                np.square(score_diff)
        summary_df['avg_samples'] = summary_df['total_samples'] / float(
            num_iterations * num_systems)
        summary_df['avg_relevant_samples'] = summary_df['total_relevant_samples'] / float(
            num_iterations * num_systems)
        summary_df['perc_within_delta'] = summary_df['num_within_delta'] / \
            float(num_iterations * num_systems)
        summary_df['perc_within_ci'] = summary_df['num_within_ci'] / \
            float(num_iterations * num_systems)
        summary_df['avg_score_diff'] = summary_df['total_score_diff'] / \
            float(num_iterations * num_systems)
        # Use custom function compute_var
        summary_df['var_score_diff'] = summary_df.apply(compute_var, axis=1,
                                                        args=[num_iterations * num_systems])

        # reorder columns, drop sampler_str column since not used
        summary_out_df = summary_df[['sampler_type', "sampler_category", 'stratification_type',
                                     "bin_style",
                                     "uses_multinomial", "avg_relevant_samples", "avg_samples",
                                     'avg_score_diff', "perc_within_ci", "perc_within_delta",
                                     'var_score_diff',
                                     "total_relevant_samples", "total_samples", "total_score_diff",
                                     "num_within_ci",
                                     "num_within_delta", "total_score_diff_sq",
                                     "metric_name", "dataset", "system_ordering"]]

        # Add "Sampler_category" and "Uses_multinomial" from summary_df to results_df for
        # Extra columns for processing
        # Drop stratification_type column since summary_df has the one we actually want
        results_df.drop(columns=['stratification_type'], inplace=True)

        results_df = results_df.merge(summary_df.loc[:, ['sampler_type', "bin_style",
                                                         "sampler_category", "uses_multinomial",
                                                         'stratification_type']],
                                      on=["sampler_type", "bin_style"])
        # Add score_diff to results df
        results_column_ordering = ['iter_num', 'sampler_type', 'sampler_name',
                                   'sampler_category', 'bin_style',
                                   'uses_multinomial',
                                   'num_step_samples', 'num_success_rounds_required', 'alpha',
                                   'delta', 'num_strata', 'stratification_type', 'exp_bin_style',
                                   'metric_object', 'exp_sampler_type', 'request_initial_samples',
                                   'initial_samples', 'total_sampled_trials',
                                   'num_rounds', 'num_init_trials', 'num_requested_init_trials',
                                   'num_nonempty_strata',
                                   'system_id', 'score', 'score_diff', "score_diff_sq",
                                   'confidence_value', 'score_variance',
                                   'sampled_trials', 'population', 'system_within_ci',
                                   'system_within_delta',
                                   'all_within_ci', 'all_within_delta',
                                   'actual_system_score',
                                   "metric_name", "dataset", "system_ordering"]
        results_out_df = results_df.loc[:, results_column_ordering]

        return summary_out_df, results_out_df

    @staticmethod
    def get_experiment_result_data(oracle_ref, input_dir, metric_object,
                                   stratification_type,
                                   num_strata, system_ordering=None,
                                   num_success_rounds_required=3,
                                   num_step_samples=100, alpha=0.05, delta=0.01,
                                   num_iterations=1000, use_initial_df=True,
                                   request_initial_samples=True,
                                   initial_samples=10, parallelize=True,
                                   random_seed=None):
        """
        The component of "run_experiment" that runs the data frames and returns the data frames.

        Args:
            oracle_ref (:obj:`aegis.oracle.oracle.Oracle`):
                the Oracle object to call to get additional trials
            input_dir (str): The path to the input directory of the submission
            metric_object (:obj:`aegis.acteval.metrics.Metric`):
                The reference to the metric object that specifies how the trials will be scored
            stratification_type (:obj:`aegis.acteval.strata.Strata`):
                Strata class that gives the stratification strategy and type of strata
            num_strata (int): The number of strata to have.
            system_ordering (list of str, optional): The desired ordering of systems, if the order
                is important.
                One can also give an ordering of only some of the systems to run the
                experiment on a subset of systems in the submission. [] Runs all systems in an
                arbitrary ordering. Defaults to None, which resolves to [].
            num_success_rounds_required (int, optional):
                The number of rounds where the (1-\\alpha) confidence
                interval's range is within +- $\\delta width. Defaults to 2
            num_step_samples (int, optional): the number of samples to ask for at each iteration;
                defaults to 100
            alpha (num, optional): The specified probability \\alpha. Defaults to 0.05.
            delta (num, optional): The specified interval range \\delta. Defaults to 0.01
            num_iterations (int, optional): The number of iterations to run each sampler for.
                Defaults to 1000
            use_initial_df (bool, optional):
                A variable that tells is to use an initial data frame if provided
                for initial samples.
            request_initial_samples (bool, optional):
                A boolean that determines if the experiment (and hence) the
                controller should request or supplement initial samples with initial samples
                to provide for adequate stratum and metric coverage for initial estimates.
                True by default
            initial_samples (int, optional): The number of samples to request initially.
                Defaults to 10.
            parallelize (bool, optional): A boolean determining whether to parallelize. True
                parallelizes with joblib, False runs in serial. Defaults to True.
            random_seed (int, optional): The desired random seed. Set to None if it is desired
                for the experiment to not be seeded. Defaults to None.

        Returns:
            (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame):
            The two data frames (summary_out_df, results_out_df) as a tuple so that they
            can be written to a file.

        """
        if system_ordering is None:
            system_ordering = []

        # # Experiment setup
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        num_systems = len(system_fpaths)
        dataset_name = os.path.basename(input_dir)
        sys_order_str = str(system_ordering)
        # Experimental parameters that are constant across all experiments
        my_metric = metric_object

        my_controller = aegis.acteval.controller.Controller()

        # Score the system against the key to get the accuracy
        system_accuracy_list = oracle_ref.get_actual_score_all_systems(system_fpaths,
                                                                       threshold_fpaths,
                                                                       my_metric)
        # Experimental parameters that vary
        sampler_type_list = [aegis.acteval.samplers.UniformTrialSampler,
                             aegis.acteval.samplers.UniformFixedTrialSampler,
                             aegis.acteval.samplers.ProportionalTrialSampler,
                             aegis.acteval.samplers.ProportionalFixedTrialSampler,
                             aegis.acteval.samplers.AdaptiveTrialSampler,
                             aegis.acteval.samplers.AdaptiveFixedTrialSampler,
                             aegis.acteval.samplers.RandomTrialSampler,
                             aegis.acteval.samplers.RandomFixedTrialSampler]
        sampler_category = ["Uniform", "Uniform", "Proportional", "Proportional",
                            "Adaptive", "Adaptive", "Random", "Random"]
        bin_style_types = ['equal', 'perc']
        uses_multinomial = [True, False, True, False, True, False, True, False]

        sampler_df = pd.DataFrame(sampler_type_list, columns=["sampler_type"])
        sampler_df['sampler_category'] = sampler_category
        sampler_df['uses_multinomial'] = uses_multinomial
        bin_df = pd.DataFrame(bin_style_types, columns=["bin_style"])

        # For summary data frame
        summary_df = sampler_df.assign(foo=1).merge(bin_df.assign(foo=1)).drop('foo', 1)
        summary_df['total_samples'] = 0
        summary_df['total_relevant_samples'] = 0
        summary_df['num_within_ci'] = 0
        summary_df['num_within_delta'] = 0
        summary_df['total_score_diff'] = 0
        summary_df['total_score_diff_sq'] = 0
        summary_df['sampler_str'] = summary_df["sampler_type"].apply(str)
        summary_df['metric_name'] = metric_object.get_metric_name()
        summary_df['dataset'] = dataset_name
        summary_df['system_ordering'] = sys_order_str
        summary_df['stratification_type'] = stratification_type

        # Do not feed initial files if desired to not use
        if not use_initial_df:
            init_fpath = None

        def process_iteration(iter_num, sampler_type, bin_style, stratification_type,
                              num_strata,
                              submission_dirname, sys_ordering,
                              random_state=None):
            """
            Process an experiment. Most parameters for the experiment come from the experiment file
            in which this method is defined. This method, in addition to returning summarized output
            as a 4-tuple, writes the fuller experiment summary to a text file.

            Args:
                iter_num (int): the current iteration. Used by joblib and to identify file output
                sampler_type (:obj:`abc.ABCMeta`): the sampler type for the experiment.
                bin_style (str): the bin style for the stratifications.
                stratification_type (:obj:`abc.ABCMeta`): The type of stratification used
                num_strata: the number of strata to use
                submission_dirname (str): The name of the submission directory
                sys_ordering (str): The system ordering provided
                random_state (:obj:`numpy.random.RandomState`, optional): The random state based
                    on the seed. If None, then the randomization
                    is not seeded. Defaults to None.

            Returns:
                (int, abc.ABCMeta, str, pandas.core.series.Series):
                Rows as a list of tuples that can be easily converted to a rows in a
                pandas data frame, one row per tuple.
                (iter_num, sampler_type, bin_style, report_tuple) as an expanded tuple. Returns
                one row per system for ease of processing.

            """
            # Give a new random state
            if random_state is None:
                rng = None
            else:
                # np.random.seed(seed=random_state)
                rng = np.random.RandomState(random_state)
            # Handle UniformRandom case with this preamble block
            my_sampler = sampler_type

            sampler_name = sampler_type.name
            if my_sampler == aegis.acteval.samplers.RandomTrialSampler:
                num_strata = -1
            elif my_sampler == aegis.acteval.samplers.RandomFixedTrialSampler:
                num_strata = -1

            # run experiment and write the outputs
            my_experiment = aegis.acteval.experiment. \
                ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                                 num_success_rounds_required=num_success_rounds_required,
                                 num_strata=num_strata,
                                 metric_object=metric_object,
                                 stratification_type=stratification_type,
                                 sampler_type=my_sampler,
                                 bin_style=bin_style,
                                 request_initial_samples=request_initial_samples,
                                 initial_samples=initial_samples)
            exp_report = my_controller.run(init_fpath, trial_data_fpath,
                                           system_fpaths, threshold_fpaths,
                                           oracle_ref, my_experiment, rng)
            all_within_ci = oracle_ref.all_within_ci(exp_report.system_list,
                                                     system_accuracy_list)
            all_within_delta = oracle_ref.all_within_delta(exp_report.system_list,
                                                           system_accuracy_list, delta)
            ci_list = [(aegis.oracle.oracle.OracleScript.system_within_ci(system, score), score)
                       for (system, score) in zip(exp_report.system_list, system_accuracy_list)]
            delta_list = \
                [(aegis.oracle.oracle.OracleScript.system_within_delta(system, score, delta),)
                 for (system, score) in zip(exp_report.system_list, system_accuracy_list)]
            # Take exp_report and return a tuple of items to go directly as a row in our data_frame
            report_tuple = exp_report.get_experiment_report_tuple()
            metric_object.get_metric_name()
            sec_fac_tuple = (my_experiment.metric_object.get_metric_name(), submission_dirname,
                             sys_ordering)
            systems_tuple_list = exp_report.get_systems_tuples()
            iteration_tuple = (iter_num, sampler_type, sampler_name, bin_style, all_within_ci,
                               all_within_delta)
            iteration_tuple_list = [iteration_tuple + report_tuple + system_tuple +
                                    system_within_ci + system_within_delta + sec_fac_tuple
                                    for (system_tuple, system_within_ci, system_within_delta)
                                    in zip(systems_tuple_list, ci_list, delta_list)]
            return iteration_tuple_list

        if not (random_seed is None):
            np.random.seed(seed=random_seed)

        rn_state_vec = np.random.randint(np.iinfo(np.int32).max,
                                         size=num_iterations * len(sampler_type_list) *
                                         len(bin_style_types))
        all_iter_lists = [range(1, num_iterations + 1), sampler_type_list,
                          bin_style_types]
        iteration_cp = list(itertools.product(*all_iter_lists))
        iteration_prod = [(rn,) + iter_item for (rn, iter_item)
                          in zip(rn_state_vec, iteration_cp)]
        if parallelize:
            results = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(process_iteration)(iter_num=i, sampler_type=sampler_type,
                                                  bin_style=bin_style,
                                                  stratification_type=stratification_type,
                                                  num_strata=num_strata,
                                                  submission_dirname=dataset_name,
                                                  sys_ordering=sys_order_str,
                                                  random_state=random_state)
                for random_state, i, sampler_type, bin_style in iteration_prod)
        else:
            results = []
            for random_state, i, sampler_type, bin_style in iteration_prod:
                results.append(process_iteration(iter_num=i,
                                                 sampler_type=sampler_type,
                                                 bin_style=bin_style,
                                                 stratification_type=stratification_type,
                                                 num_strata=num_strata,
                                                 submission_dirname=dataset_name,
                                                 sys_ordering=sys_order_str,
                                                 random_state=random_state))

        # Get a nice data frame
        results_header = ["iter_num", "sampler_type", "sampler_name", "bin_style", "all_within_ci",
                          "all_within_delta",
                          "num_step_samples",
                          "num_success_rounds_required", "alpha", "delta", "num_strata",
                          "stratification_type", "exp_bin_style", "metric_object",
                          "exp_sampler_type",
                          "request_initial_samples", "initial_samples",
                          "total_sampled_trials", "num_rounds",
                          "num_init_trials", "num_requested_init_trials", "num_nonempty_strata",
                          "system_id", "score", "confidence_value", "score_variance",
                          "sampled_trials", "population", "system_within_ci",
                          "actual_system_score", "system_within_delta",
                          "metric_name", "dataset", "system_ordering"]
        # Using
        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
        # to flatten lists
        results_rows = []
        for sublist in results:
            for item in sublist:
                results_rows.append(item)
        results_df = pd.DataFrame(results_rows, columns=results_header)
        results_df['score_diff'] = results_df['score'] - \
            results_df['actual_system_score']
        results_df['score_diff_sq'] = results_df['score_diff'].apply(np.square)

        for index, row in results_df.iterrows():
            # Check within delta not within the system's estimate of the CI for the summary
            sampler_str, bin_style, all_within_ci, all_within_delta, sampled_trials, score_diff, \
                total_sampled_trials = \
                str(row["exp_sampler_type"]), row["bin_style"], row["all_within_ci"], \
                row["all_within_delta"], row["sampled_trials"],  row["score_diff"], \
                row["total_sampled_trials"]
            if all_within_delta:
                summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                               (summary_df['bin_style'] == bin_style), 'num_within_delta'] += 1
            if all_within_ci:
                summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                               (summary_df['bin_style'] == bin_style), 'num_within_ci'] += 1
            summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                           (summary_df['bin_style'] == bin_style), 'total_relevant_samples'] += \
                sampled_trials
            summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                           (summary_df['bin_style'] == bin_style), 'total_samples'] += \
                total_sampled_trials
            # For score differences take all of the systems
            # and then divide by num_iterations*num_systems
            summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                           (summary_df['bin_style'] == bin_style), 'total_score_diff'] += \
                score_diff
            summary_df.loc[(summary_df['sampler_str'] == sampler_str) &
                           (summary_df['bin_style'] == bin_style), 'total_score_diff_sq'] += \
                np.square(score_diff)

        summary_df['avg_samples'] = summary_df['total_samples'] / float(num_iterations*num_systems)
        summary_df['avg_relevant_samples'] = summary_df['total_relevant_samples'] / float(
            num_iterations * num_systems)
        summary_df['perc_within_delta'] = summary_df['num_within_delta'] / \
            float(num_iterations*num_systems)
        summary_df['perc_within_ci'] = summary_df['num_within_ci'] / \
            float(num_iterations * num_systems)
        summary_df['avg_score_diff'] = summary_df['total_score_diff'] / \
            float(num_iterations*num_systems)
        # Use custom function compute_var
        summary_df['var_score_diff'] = summary_df.apply(compute_var, axis=1,
                                                        args=[num_iterations*num_systems])

        # reorder columns, drop sampler_str column since not used
        summary_out_df = summary_df[['sampler_type', "sampler_category", 'stratification_type',
                                     "bin_style",
                                     "uses_multinomial", "avg_relevant_samples", "avg_samples",
                                     'avg_score_diff', "perc_within_ci", "perc_within_delta",
                                     'var_score_diff',
                                     "total_relevant_samples", "total_samples", "total_score_diff",
                                     "num_within_ci",
                                     "num_within_delta", "total_score_diff_sq",
                                     "metric_name", "dataset", "system_ordering"]]

        # Add "Sampler_category" and "Uses_multinomial" from summary_df to results_df for
        # Extra columns for processing
        # Drop stratification_type column since summary_df has the one we actually want
        results_df.drop(columns=['stratification_type'], inplace=True)

        results_df = results_df.merge(summary_df.loc[:, ['sampler_type', "bin_style",
                                                         "sampler_category", "uses_multinomial",
                                                         'stratification_type']],
                                      on=["sampler_type", "bin_style"])
        # Add score_diff to results df
        results_column_ordering = ['iter_num', 'sampler_type', 'sampler_name',
                                   'sampler_category', 'bin_style',
                                   'uses_multinomial',
                                   'num_step_samples', 'num_success_rounds_required', 'alpha',
                                   'delta', 'num_strata', 'stratification_type', 'exp_bin_style',
                                   'metric_object', 'exp_sampler_type', 'request_initial_samples',
                                   'initial_samples', 'total_sampled_trials',
                                   'num_rounds', 'num_init_trials', 'num_requested_init_trials',
                                   'num_nonempty_strata',
                                   'system_id', 'score', 'score_diff', "score_diff_sq",
                                   'confidence_value', 'score_variance',
                                   'sampled_trials', 'population', 'system_within_ci',
                                   'system_within_delta',
                                   'all_within_ci', 'all_within_delta',
                                   'actual_system_score',
                                   "metric_name", "dataset", "system_ordering"]
        results_out_df = results_df.loc[:, results_column_ordering]

        return summary_out_df, results_out_df

    @staticmethod
    def run_experiment(oracle_ref, input_dir, output_dir, metric_object, stratification_type,
                       num_strata, system_ordering=None,
                       num_success_rounds_required=3,
                       num_step_samples=100, alpha=0.05, delta=0.01,
                       num_iterations=1000, use_initial_df=True, request_initial_samples=True,
                       initial_samples=40, parallelize=True,
                       run_id="experiment_0",
                       batch_id=datetime.datetime.now().strftime('%Y-%m-%d'),
                       git_commit_hash="No Git Commit Hash Provided",
                       random_seed=None):
        """
        Runs an "experiment", which takes the submission with the specified metric, stratification
        type and experimental parameters, and runs on all samplers and with all strataficiation bin
        styles, running each experiment as specified. Calls
        :func:`aegis.oracle.oracle.OracleScript.get_experiment_result_data()`

        Args:
            oracle_ref (:obj:`aegis.oracle.oracle.Oracle`):
                the Oracle object to call to get additional trials
            input_dir (str): The path to the input directory of the submission
            output_dir (str): The path to the output directory to create the results in.
            metric_object (:obj:`aegis.acteval.metrics.Metric`):
                The reference to the metric object that specifies how the trials will be scored
            stratification_type (:obj:`aegis.acteval.strata.Strata`):
                Strata class that gives the stratification strategy and type of strata
            num_strata (int): The number of strata to have.
            system_ordering (list of str): The desired ordering of systems, if the order
                is important. Order is specified by system id.
                None, which resolves to [] By
                default. One can also give an ordering of only some of the systems to run the
                experiment on a subset of systems in the submission. [] Runs all systems in an
                arbitrary ordering.
            num_success_rounds_required (int, optional):
                The number of rounds where the (1-\\alpha) confidence
                interval's range is within +- $\\delta width. Defaults to 2
            num_step_samples (int, optional): the number of samples to ask for at each iteration
                defaults to 100
            alpha (num, optional): The specified probability \\alpha. Defaults to 0.05.
            delta (num, optional): The specified interval range \\delta. Defaults to 0.01
            num_iterations (int, optional): The number of iterations to run each sampler for.
                Defaults to 1000
            use_initial_df (bool, optional): A variable that tells is to use an initial
                data frame if provided
                for initial samples. Defaults to True.
            request_initial_samples (bool, optional):
                A boolean that determines if the experiment (and hence) the
                controller should request or supplement initial samples with initial samples
                to provide for adequate stratum and metric coverage for initial estimates.
                True by default.
            initial_samples (int, optional): The number of samples to request initially.
                Defaults to 100.
            parallelize (bool, optional): A boolean determining whether to parallelize. True
                parallelizes with joblib, False runs in serial. Defaults to True.
            run_id (str, optional): An optional run_id value to add to the date.
                Default "experiment_0", but recommended
                to be a combination of the submission name and the experiment parameters.
            batch_id (str, optional): An optional batch_id value to add, the date by default
            git_commit_hash (str, optional): The string of the git commit hash of this code.
                If provided, it will write it to a text file git_commit_hash.txt.
                Default value is "No Git Commit Hash Provided" so that we have something
                to a file.
            random_seed (int, optional): The desired random seed. Set to None if it is desired
                for the experiment to not be seeded. Defaults to None.

        Returns:
            Nothing. Directory Output_dir is created if it does not exist and additional
            files from the experiment (including subfolders) are created. Files existing with the
            same name as created files are overwritten, but files of other names are not deleted.

        """

        if system_ordering is None:
            system_ordering = []

        logger = logging.getLogger('paper_experiments_logger')
        for hdlr in logger.handlers[:]:  # remove all old handlers
            logger.removeHandler(hdlr)
        logfile_fpath = os.path.join(output_dir, str(batch_id) + '.log')
        logger.addHandler(logging.FileHandler(logfile_fpath))
        logger.info(f'batch_id: {batch_id}')
        logger.info(f'input_dir: {input_dir}')
        logger.info(f'run_id: {run_id}')
        logger.info(f'random_seed: {random_seed}')
        stratification_type_string = str(stratification_type).split('\'')[1].split('.')[-1]
        system_ordering_string = '_'.join(system_ordering)
        logger.info(f'stratification_type_string: {stratification_type_string}')
        logger.info(f'system_ordering_string: {system_ordering_string}')
        try:
            # Organize runs by batch and then by run
            exp_output_dir = os.path.join(output_dir, batch_id, run_id)
            if not os.path.exists(exp_output_dir):
                os.makedirs(exp_output_dir)

            # Write the git commit hash as a text file
            git_commit_hash_file = os.path.join(exp_output_dir, "git_commit_hash.txt")
            git_fobj = open(git_commit_hash_file, "w")
            git_fobj.write(git_commit_hash)
            git_fobj.close()

            # Now write the experimental parameters to a file
            exp_params_file = os.path.join(exp_output_dir, "experimental_params.txt")
            exp_fobj = open(exp_params_file, "w")
            exp_fobj.write("batch_id: " + str(batch_id))
            exp_fobj.write("\nrun_id: " + str(run_id))
            exp_fobj.write("\ninput_dir: " + str(input_dir))
            exp_fobj.write("\nsystem ordering: " + str(system_ordering))
            exp_fobj.write("\nrandom seed: " + str(random_seed) + "\n")
            exp_obj = aegis.acteval.experiment. \
                ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                                 num_success_rounds_required=num_success_rounds_required,
                                 num_strata=num_strata,
                                 metric_object=metric_object,
                                 stratification_type=stratification_type,
                                 sampler_type="\'<Varying Experimental Factor>\'",
                                 bin_style="\'<Varying Experimental Factor>\'",
                                 request_initial_samples=request_initial_samples,
                                 initial_samples=initial_samples)
            exp_fobj.write(str(exp_obj))
            exp_fobj.close()

            (summary_out_df, results_out_df) = aegis.oracle.oracle.OracleScript. \
                get_experiment_result_data(oracle_ref,
                                           input_dir=input_dir,
                                           metric_object=metric_object,
                                           stratification_type=stratification_type,
                                           num_strata=num_strata,
                                           system_ordering=system_ordering,
                                           num_success_rounds_required=num_success_rounds_required,
                                           num_step_samples=num_step_samples, alpha=alpha,
                                           delta=delta,
                                           num_iterations=num_iterations,
                                           use_initial_df=use_initial_df,
                                           request_initial_samples=request_initial_samples,
                                           initial_samples=initial_samples,
                                           parallelize=parallelize,
                                           random_seed=random_seed)

            summary_out_df.to_csv(os.path.join(exp_output_dir, "summary_of_results.csv"),
                                  index=False,
                                  header=True)
            results_out_df.to_csv(os.path.join(exp_output_dir, "individual_run_results.csv"),
                                  index=False, header=True)
        except Exception as e:
            print(e)
            logger.fatal(e, exc_info=True)  # log exception info at FATAL log level

        except Warning as w:
            print(w)
            logger.warning(w, exec_info=True)

        tmp_files_list = glob.glob('./*.tmp')
        outfile = open(logfile_fpath, 'a')
        for fname in tmp_files_list:
            with open(fname) as infile:
                outfile.write(infile.read())

        for my_fname in tmp_files_list:
            os.remove(my_fname)

    @staticmethod
    def run_specific_experiment(oracle_ref, input_dir, output_dir, metric_object,
                                stratification_type,
                                num_strata, system_ordering=None, sampler_types=None,
                                sampler_categories=None,
                                bin_style_types=None,
                                use_multinomials=None,
                                num_success_rounds_required=3,
                                num_step_samples=100, alpha=0.05, delta=0.01,
                                num_iterations=1000, use_initial_df=True,
                                request_initial_samples=True,
                                initial_samples=40, parallelize=True,
                                run_id="experiment_0",
                                batch_id=datetime.datetime.now().strftime('%Y-%m-%d'),
                                git_commit_hash="No Git Commit Hash Provided",
                                random_seed=None):
        """
        Runs a specified "experiment", which takes the submission with the specified metric,
        stratification
        type and experimental parameters, and runs on samplers and with strataficiation bin
        styles as specified, running each experiment as specified. Calls
        :func:`aegis.oracle.oracle.OracleScript.get_specific_experiment_result_data()`

        Args:
            oracle_ref (:obj:`aegis.oracle.oracle.Oracle`):
                the Oracle object to call to get additional trials
            input_dir (str): The path to the input directory of the submission
            output_dir (str): The path to the output directory to create the results in.
            metric_object (:obj:`aegis.acteval.metrics.Metric`):
                The reference to the metric object that specifies how the trials will be scored
            stratification_type (:obj:`aegis.acteval.strata.Strata`):
                Strata class that gives the stratification strategy and type of strata
            num_strata (int): The number of strata to have.
            system_ordering (list of str): The desired ordering of systems, if the order
                is important. Order is specified by system id.
                None, which resolves to [] By
                default. One can also give an ordering of only some of the systems to run the
                experiment on a subset of systems in the submission. [] Runs all systems in an
                arbitrary ordering.
            sampler_types (list of :obj:`abc.ABCMeta`, optional):
                List of samplers you wish to use in experiment
            sampler_categories (list of str, optional): List of sampler categories you
                want to use in experiment. Defaults to None, which resolves as [].
            bin_style_types (list of str, optional): List of bin style types you want to
                use in experiment. Defaults to None, which resolves as [].
            use_multinomials (list of bool, optional): List of booleans for each sampler type
                Defaults to None, which resolves as [].
            num_success_rounds_required (int, optional):
                The number of rounds where the (1-\\alpha) confidence
                interval's range is within +- $\\delta width. Defaults to 2
            num_step_samples (int, optional): the number of samples to ask for at each iteration
                defaults to 100
            alpha (num, optional): The specified probability \\alpha. Defaults to 0.05.
            delta (num, optional): The specified interval range \\delta. Defaults to 0.01
            num_iterations (int, optional): The number of iterations to run each sampler for.
                Defaults to 1000
            use_initial_df (bool, optional): A variable that tells is to use an initial
                data frame if provided
                for initial samples. Defaults to True.
            request_initial_samples (bool, optional):
                A boolean that determines if the experiment (and hence) the
                controller should request or supplement initial samples with initial samples
                to provide for adequate stratum and metric coverage for initial estimates.
                True by default.
            initial_samples (int, optional): The number of samples to request initially.
                Defaults to 100.
            parallelize (bool, optional): A boolean determining whether to parallelize. True
                parallelizes with joblib, False runs in serial. Defaults to True
            run_id (str, optional): An optional run_id value to add to the date.
                Default "experiment_0", but recommended
                to be a combination of the submission name and the experiment parameters.
            batch_id (str, optional): An optional batch_id value to add, the date by default
            git_commit_hash (str, optional): The string of the git commit hash of this code.
                If provided, it will write it to a text file git_commit_hash.txt.
                Default value is "No Git Commit Hash Provided" so that we have something
                to a file.
            random_seed (int, optional): The desired random seed. Set to None if it is desired
                for the experiment to not be seeded. Defaults to None.

        Returns:
            Nothing. Directory Output_dir is created if it does not exist and additional
            files from the experiment (including subfolders) are created. Files existing with the
            same name as created files are overwritten, but files of other names are not deleted.

        """

        if system_ordering is None:
            system_ordering = []

        if sampler_types is None:
            sampler_types = []

        if sampler_categories is None:
            sampler_categories = []

        if bin_style_types is None:
            bin_style_types = []

        if use_multinomials is None:
            use_multinomials = []

        # Organize runs by batch and then by run
        logger = logging.getLogger('paper_experiments_logger')
        for hdlr in logger.handlers[:]:  # remove all old handlers
            logger.removeHandler(hdlr)
        logfile_fpath = os.path.join(output_dir, str(batch_id) + '.log')
        logger.addHandler(logging.FileHandler(logfile_fpath))
        logger.info(f'batch_id: {batch_id}')
        logger.info(f'input_dir: {input_dir}')
        logger.info(f'run_id: {run_id}')
        logger.info(f'random_seed: {random_seed}')
        stratification_type_string = str(stratification_type).split('\'')[1].split('.')[-1]
        system_ordering_string = '_'.join(system_ordering)
        logger.info(f'stratification_type_string: {stratification_type_string}')
        logger.info(f'system_ordering_string: {system_ordering_string}')
        try:
            exp_output_dir = os.path.join(output_dir, batch_id, run_id)
            if not os.path.exists(exp_output_dir):
                os.makedirs(exp_output_dir)

            # Write the git commit hash as a text file
            git_commit_hash_file = os.path.join(exp_output_dir, "git_commit_hash.txt")
            git_fobj = open(git_commit_hash_file, "w")
            git_fobj.write(git_commit_hash)
            git_fobj.close()

            # Now write the experimental parameters to a file
            exp_params_file = os.path.join(exp_output_dir, "experimental_params.txt")
            exp_fobj = open(exp_params_file, "w")
            exp_fobj.write("batch_id: " + str(batch_id))
            exp_fobj.write("\nrun_id: " + str(run_id))
            exp_fobj.write("\ninput_dir: " + str(input_dir))
            exp_fobj.write("\nsystem ordering: " + str(system_ordering))
            exp_fobj.write("\nrandom seed: " + str(random_seed) + "\n")
            exp_obj = aegis.acteval.experiment. \
                ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                                 num_success_rounds_required=num_success_rounds_required,
                                 num_strata=num_strata,
                                 metric_object=metric_object,
                                 stratification_type=stratification_type,
                                 sampler_type="\'<Varying Experimental Factor>\'",
                                 bin_style="\'<Varying Experimental Factor>\'",
                                 request_initial_samples=request_initial_samples,
                                 initial_samples=initial_samples)
            exp_fobj.write(str(exp_obj))
            exp_fobj.close()

            (summary_out_df, results_out_df) = aegis.oracle.oracle.OracleScript. \
                get_specific_experiment_result_data(
                oracle_ref,
                sampler_types=sampler_types,
                sampler_categories=sampler_categories,
                bin_style_types=bin_style_types,
                use_multinomials=use_multinomials,
                input_dir=input_dir,
                metric_object=metric_object,
                stratification_type=stratification_type,
                num_strata=num_strata,
                system_ordering=system_ordering,
                num_success_rounds_required=num_success_rounds_required,
                num_step_samples=num_step_samples, alpha=alpha,
                delta=delta,
                num_iterations=num_iterations,
                use_initial_df=use_initial_df,
                request_initial_samples=request_initial_samples,
                initial_samples=initial_samples,
                parallelize=parallelize,
                random_seed=random_seed)

            summary_out_df.to_csv(os.path.join(exp_output_dir, "summary_of_results.csv"),
                                  index=False,
                                  header=True)
            results_out_df.to_csv(os.path.join(exp_output_dir, "individual_run_results.csv"),
                                  index=False, header=True)
        except Exception as e:
            print(e)
            logger.fatal(e, exc_info=True)  # log exception info at FATAL log level
        except Warning as w:
            print(w)
            logger.warning(w, exec_info=True)

        tmp_files_list = glob.glob('./*.tmp')
        outfile = open(logfile_fpath, 'a')
        for fname in tmp_files_list:
            with open(fname) as infile:
                outfile.write(infile.read())

        for my_fname in tmp_files_list:
            os.remove(my_fname)
