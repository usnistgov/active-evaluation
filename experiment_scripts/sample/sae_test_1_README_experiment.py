#! /usr/bin/env python

import git
import datetime
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.oracle.oracle
import aegis.acteval.experiment

# Get and store the current git commit hash for reference
repo = git.Repo(search_parent_directories=True)
git_commit_hash = repo.head.object.hexsha

random_seed = 936712
num_success_rounds_required = 3
num_step_samples = 100
alpha = 0.10
delta = 0.20
num_iterations = 3
use_initial_df = False
request_initial_samples = True
initial_samples = 200
output_dir = "../../readme_experiment_outputs"
curr_datetime = datetime.datetime.now()

# Specify the Data Directory with the Batch
batch_id = str(curr_datetime.strftime('%Y-%m-%d')) + "_README_experiments"
key_fpath = "../../data/test/sae_test_1/key.csv"
input_dir = "../../data/test/sae_test_1"

# First Run: Multi-System experiment
run_num = "README_multi_sae_test_1_s1s2"
system_ordering = ["s1", "s2"]
num_strata = 4
metric_object = aegis.acteval.metrics.BinaryAccuracyMetric(key_values=[0, 1])
stratification_type = aegis.acteval.strata.StrataMultiSystemIntersectDecision
run_id = run_num + "_" + str(random_seed)
# Run the Experiment
oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)


aegis.oracle. \
    oracle.OracleScript.run_experiment(oracle_ref,
                                       input_dir=input_dir, output_dir=output_dir,
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
                                       parallelize=False, run_id=run_id, batch_id=batch_id,
                                       git_commit_hash=git_commit_hash,
                                       random_seed=random_seed)


# Second Run: Single-System Experiment
run_num = "README_single_sae_test_1_s1"
system_ordering = ["s1"]
num_strata = 4
metric_object = aegis.acteval.metrics.BinaryAccuracyMetric(key_values=[0, 1])
stratification_type = aegis.acteval.strata.StrataFirstSystemDecision
run_id = run_num + "_" + str(random_seed)
# Run the Experiment
oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

aegis.oracle. \
    oracle.OracleScript.run_experiment(oracle_ref,
                                       input_dir=input_dir, output_dir=output_dir,
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
                                       parallelize=False, run_id=run_id, batch_id=batch_id,
                                       git_commit_hash=git_commit_hash,
                                       random_seed=random_seed)
