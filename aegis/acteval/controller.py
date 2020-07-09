import numpy as np
import math
import aegis.acteval.samplers
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.data_processor
import aegis.acteval.summary_reporter
import aegis.acteval.experiment
import logging


class Controller:
    """
    Class to represent the controller that handles the logic that runs the active evaluation.
    It takes in experimental parameters and an oracle, and returns a summary report.
    """

    def __init__(self):
        """
        Construct a controller.

        The controller can run different types of experiments, so the run() method takes the
        experiment parameters.
        """

        logger = logging.getLogger('paper_experiments_logger')
        logger.info('Initiating Controller class')

        pass

    def run(self, init_fpath, trial_data_filepath, system_fpaths,
            threshold_fpaths, oracle_ref,
            my_experiment=aegis.acteval.experiment.ExperimentParams(),
            rng=None, total_runs=math.inf):
        """
        Run the controller to simulate an experiment.

        The controller will call the referred to Oracle when needed to get trial samples.

        Args:
            init_fpath (str): the filepath to the pointed data frame of
                initially sampled trials with ground truth. None if no initial samples exist
            trial_data_filepath (str): the filepath to the trial metadata or features (the data)
            system_fpaths (list of object): the list of filepaths to the system output files
            threshold_fpaths (list of str):
                the list of filepaths corresponding to the system thresholds for the metrics
            oracle_ref (:obj:`aegis.oracle.oracle.Oracle`):
                the Oracle object to call to get additional trials
            my_experiment (:obj:`aegis.acteval.experiment.ExperimentParams`):
                The Experiment Object with all of the experimental parameters.
            rng (:obj:`numpy.random.RandomState`, optional): The random state that is used as the
                random number generator. If none, we will generate one in the controller.
                rng is generated or passed here so that randomization still works even
                for parallel implementations. Defaults to None.
            total_runs (int, optional): Integer to tell how many runs you want to do, defaults
                to math.inf which means to run until we have my_experiment.num_successful_rounds
                successful runs. Defaults to `math.inf`.

        Returns:
            aegis.acteval.summary_reporter.SummaryReport: summary_report,
            a summary report of the experiment that can be printed to the
            screen or any output stream

        """
        # Initialization
        if rng is None:
            rng = np.random.RandomState()

        num_previous_successes = 0
        num_rounds = 0

        # start up logger
        logger = logging.getLogger("paper_experiments_logger"+"." +
                                   str(my_experiment.stratification_type) +
                                   "." + str(my_experiment.sampler_type) +
                                   "." + str(my_experiment.bin_style))
        # logger.info("Run method of the controller class")

        # get values from Experiment object
        num_step_samples = my_experiment.num_step_samples
        num_success_rounds_required = my_experiment.num_success_rounds_required

        delta = my_experiment.delta
        num_strata = my_experiment.num_strata
        strata_type = my_experiment.stratification_type
        bin_style = my_experiment.bin_style
        metric_obj = my_experiment.metric_object
        sampler_type = my_experiment.sampler_type

        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_df = None
        if not (init_fpath is None):
            init_df = my_data_processor.process_init_data(init_fpath)
        # trial_df = my_data_processor.process_trial_data(trial_data_filepath)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        # We must obtain decisions before we stratify for those methods that incorporate
        # the decision into the stratification, as well as before we construct the strata
        [metric_obj.convert_thresholds_to_decisions(system) for system in system_list]

        my_strata = strata_type(num_strata, system_list)
        my_strata.dirty_strata_cache()
        # Pick alpha based on stratification method to adjust for the proper number of Type I
        # alpha
        strata_alpha = my_strata.get_strata_alpha(my_experiment.alpha)
        # Shrink alpha in computations to dialate confidence interval
        alpha_shrinkage = 0.9
        alpha = strata_alpha * alpha_shrinkage

        my_strata.stratify(bin_style)

        num_total_samples = 0
        if not (init_df is None):
            num_total_samples = init_df.shape[0]
            # In Initialization, Set first samples but do not do a successful round check
            my_strata.add_samples_to_strata(init_df)

        num_requested_init_samples = 0
        # Supplement initial samples with additional samples if asked in experiment
        if my_experiment.request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_obj,
                                                      my_experiment.initial_samples, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_total_samples += len(initial_samples)
            num_requested_init_samples += len(initial_samples)

        # We need to evaluate the score after the initial samples
        my_strata.estimate_samples_all_systems(metric_obj)
        my_strata.estimate_pop_all_systems(metric_obj)
        my_strata.estimate_score_all_systems(metric_obj)
        my_strata.estimate_score_variance_all_systems(metric_obj)
        my_strata.estimate_pop_variance_all_systems(metric_obj)
        my_strata.estimate_score_variance_upper_all_systems(metric_obj, alpha)
        metric_obj.estimate_population_intervals_all_systems_strata(my_strata, alpha)
        my_strata.estimate_score_lower_all_systems(metric_obj, alpha)
        my_strata.estimate_score_upper_all_systems(metric_obj, alpha)
        my_strata.get_confidence_intervals_all_systems(metric_obj, alpha)

        trial_sampler = sampler_type(my_strata, num_success_rounds_required)
        while trial_sampler.sample_next_round(num_previous_successes) and num_rounds < total_runs:
            # samples here is a list of trial_id values
            samples = trial_sampler.draw_samples(
                num_step_samples, metric_obj, rng=rng, alpha=alpha
            )
            num_total_samples += len(samples)
            annotations_df = oracle_ref.get_annotations(samples)
            my_strata.add_samples_to_strata(annotations_df)
            # Update stratum and system information; return values are not used and
            # thus discarded
            # By calling these methods now, we can refer to the stratum objects to get these
            # lists rather than re-computing
            my_strata.estimate_samples_all_systems(metric_obj)
            my_strata.estimate_pop_all_systems(metric_obj)
            my_strata.estimate_score_all_systems(metric_obj)
            my_strata.estimate_score_variance_all_systems(metric_obj)
            my_strata.estimate_pop_variance_all_systems(metric_obj)
            my_strata.estimate_score_variance_upper_all_systems(metric_obj, alpha)
            metric_obj.estimate_population_intervals_all_systems_strata(my_strata, alpha)
            my_strata.estimate_score_lower_all_systems(metric_obj, alpha)
            my_strata.estimate_score_upper_all_systems(metric_obj, alpha)
            my_strata.get_confidence_intervals_all_systems(metric_obj, alpha)
            succ_round = trial_sampler.meets_confidence_criteria(
                my_strata, delta, alpha, metric_obj
            )
            if succ_round and (len(samples) == 0):
                num_previous_successes += 1
                print("Although a successful round, no new samples selected this round in round " +
                      str(num_rounds))
                logger.info("Although a successful round, no new samples selected this round in"
                            " round " + str(num_rounds))
            elif succ_round:
                num_previous_successes += 1
            elif len(samples) == 0:
                # If we have no new samples at all, mark the round as successful with a footnote
                print("No new samples selected this round in round " +
                      str(num_rounds) + ": marking round as successful")
                logger.info("No new samples selected this round in round " +
                            str(num_rounds) + ": marking round as successful")
                num_previous_successes += 1
            else:
                num_previous_successes = 0
            num_rounds += 1

        # Post results
        num_samples_per_stratum = [
            sum(st.estimate_samples_all_systems(metric_obj)) for st in my_strata.strata]
        # num_samples_per_stratum = my_strata[0].estimate_samples
        init_trials = 0
        if not (init_df is None):
            init_trials = init_df.shape[0]
        summary_report = aegis.acteval.summary_reporter.SummaryReport(my_experiment,
                                                                      my_strata.system_list,
                                                                      num_rounds,
                                                                      init_trials,
                                                                      num_total_samples,
                                                                      num_requested_init_samples,
                                                                      my_strata.num_strata,
                                                                      num_samples_per_stratum)
        return summary_report
