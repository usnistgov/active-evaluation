# CHANGELOG

## v 2020.03.03

Updated documentation and added first working Recall Metric. Also:

* added uncertainties for population measurements, including population variances,
upper population estimates, population confidence intervals, and updated samplers and confidence estimates to leverage the upper population estimates when needed to compute more accurate confidence intervals.
* Upper population uncertainty set to 0 for all Accuracy and Precision Metrics
* Upper population uncertainty for recall is 0 if we have at least one relevant sample.
Else, we estimate the uncertainty of the recall population
* Implemented Sampler Classes `TrueRandomTrialSampler` and `TrueRandomFixedTrialSampler` that sample
from all trials, including potentially non-relevant ones. `RandomTrialSampler` and `RandomFixedTrialSampler`
now sample only from relelvant trials for Precision.
* Updated the README
* Streamlined and Updated the Code Documentation, checking and updating all python docstrings
of aegis.acteval and aegis.oracle
* Updated logging to work within the Oracle Class allowing for smoother experiments
* Removed obsolete and incomplete metric class `BinaryAccuracyTwoProportionMetric()`
* Eliminated mutable default arguments of [] and replaced with None (assigning to []) within the
methods to prevent possible bugs.
* Changed experiment scripts to be able to be run in subfolders
rather than the root directory
* Organized experiment scripts into subfolders
* Updated setup.py for streamlined package installation
that works with recent versions of pip


## v 2019.09.25

* Changed the score tuning for Precision and Recall to now be adjusted (decreased) by multiplying it by the finite population correction.
* Changed Precision initialization to find different samples relevant for each specific system, rather than just a set of samples that is relevant for the combination of systems.
* Added logging capabilities to generate logs during experiments for error troubleshooting

## v 2019.08.30

* Added specific classes `RandomTrialSampler` and `RandomFixedTrialSampler` to perform the random
sampling rather than altering the stratification for UniformFixedTrialSampler. This allows us
to see differences in the initial sampling and the stratification.
* Changed `initial_samples_per_bin` to `initial_samples` in order to make experiments more consistent
* Updated experiments to have more initial samples (since the initial samples are now a total and
not just per bin or per stratum).
* Added First implementation of experiment logging
* Added Paper 1 experiment files covering Math1 and OpenML Datasets for Aegis experiment work.
* Fixed bugs with Precision sampling and marking rounds as successful when there are no additional
samples drawn.

## v 2019.08.12

Factor-of-10 speedup in code and bug fixes

* `get_combined_systems_df()` and `get_combined_systems_score_df()` methods cached to not redundantly compute. 
Caching occurs at the strata and stratum levels.
* `get_combined_systems_df()`,`get_combined_systems_score_df()`, and `get_combined_data_frames()` optimized to further speed up performance. (A Profiler was run to check this)
* Adjusted Bugs in Random seeding
* Fixed bugs in `BinaryAccuracyTwoProportionMetric` score variance computation.
* Improved speed of computation in `BinaryAccuracyTwoProportionMetric` score computation.
* Added `Math1` datasets and submissions with system outputs
* Added OpenML submissions for the `6_letter` and the `151_electricity` datasets. These include
the outputs of 5 ML systems
* Added experimental scripts to run paper experiments.
* Fixed bug in `BinaryPrecisionMetric` to support positive labels other than `1`.
* Added factors and factor labels in experiment output data frames for ease of analysis and post-processing.

## v 2019.07.10

Enhancements to the Precision metric implementation and bugfixes.

## v 2019.07.02

Implementation of accuracy as two proportions and implemented support for the Precision Metric.

* Implemented Metric subclasses `BinaryAccuracyTwoProportionMetric` and `BinaryPrecisionMetric`
* Fixed bugs with running experiments in `StrataFirstSystemDecision` so experiments finish without crashing.
* Fixed `draw_samples` to only draw from relevant random samples in Uniform, Proportional, and Adaptive Samplers. (Useful for precision)

## v 2019.06.10


Implementation of Stratification that also accounts for the system decision.

* Implementation of Strata subclasses `StrataFirstSystemDecision` and `StrataMultiSystemIntersectDecision`
* Deleted unused method `estimate_variance` from all Metric objects.
* Adjusted tuning for Accuracy Metrics to improve confidence intervals.

Bugfixes and performance enhancements.

## v 2019.05.31

Improved calling of random number generator to work with `joblib`. Additional changes:

* Added experimental_params.txt file that writes all of the experimental parameters to a text file
for ease of reference.
* Fixed a division by 0 bug and added a test case
* Uses `numpy.random.RandomState` for seeds

## v 2019.05.31

Added corrections for estimates for single- and multi-system experiments.

* Multi-system experiments now correct for type-I error by using a union bound so that all systems
are within the CI `1 - alpha` of the time
* Added a correction to the score estimate to reduce bias
* Added finite population corrections to any estimate corrections (for the score estimate correction
and the score variance estimate correction)
* Changed the width of score estimate corrections to be proportional to the square root of the
number of samples (mimicking m-estimate smoothing)
* Changed run_experiment() to have `run_id` and `batch_id` parameters for streamlined organization and easier tracking of experiments.

## v 2019.05.22

Added additional experiment features and changed float computations from `math` library
to `numpy` library. This means that the version of numpy is important for replication.

* git_commit_hash.txt gives commit hash of aegis version used for experiment
* Response variable of (estimated score - actual score), `avg_score_diff` added to summary df.
Variance of score difference also included.
* score difference computations added to individual runs data frame. Squared score difference 
also included.

## v 2019.05.20

Changed computations by lowering internal prior_samples_correction to 0.2

* Enhanced data frame rows to contain additional information
* Enhanced data frame run rows splitting samplers to be categorized among three dimensions: 
sampler_category, bin_style, and uses_multinomial
* Changed StrataMultiSystemIntersect to use the max to aggregate stats instead of the mean;
this results in sampling logic trying to sample to reduce maximum score variance.

## v 2019.05.09

Added experiment running methods for ease of running experiments

* In addition to a summary table, now each run is given as a row in a data frame with fields of
important information, for ease of plotting and anaysis
* Class OracleScript() has method run_experiment() that runs experiments and provides an easy API
* An executable script for running single-system experiments and multi-system experiments are 
updated.

## v 2019.05.06

Adjusted confidence intervals for correctness

* Used Agresti Approximation for score variance that adds in 2 true positives and 2
false positives to score variance to improve ci coverage
* Added new metric BinaryAccuracyTwoProportionMetric that models accuracy not as a
single proportion but as two proportions: one over all examples where the system considers
an example to be the low key value, and the other over the remaining examples, which are those
that the system classifies as a high key value.
* added method find_initial_samples() in the strata object that allows the system to give
an initialization that minimially covers all stratum in the strata. This method can be used
to provide an init_df or supplement an init_df, with the hope of reducing the number of rounds
needed for sampling.

## v 2019.05.03

Test Cases for computational correctness and stratification correctness, and bugfixes to
improve the math.

Additional updates:

* Fixing of code to eliminate divisions by 0, no longer triggering numpy warning `RuntimeWarning: invalid value encountered in double_scalars for i in self.strata.strata`



## v 2019.04.29

Per-iteration parallelization of experiment files to use multiple cores for faster computation.
Uses the `joblib` package of python.

## v 2019.04.25

Implementation of multi-system stratification that involves all systems.

Additional Updates:
* Multi-system stratification implemented with the class StrataMultiSystemIntersect. This
stratifies each system into so many bins, and then intersects these bins to form disjoint
strata. This often produces some empty strata.
* __repr__ method converted to __str__ methods since they were pretty-print methods
* aegis.acteval.experiment.ExperimentalParams object added. It functions as a struct for Experimental Parameters
* Summary report updated to take in experiment and System objects
* __str__ methods updated for System, ExperimentParam, and SummaryReport objects to give useful information when printing
* OracleScript updated to provide methods to compute system scores on the full key and to determine if the system's score estimate confidence interval contains the true score.
* Added finite population correction in the strata computation of score variance. This is 
necessary to handle cases when we have exhausted a stratum
* Handled cases where we exhausted a stratum to sample what is left and then carry on. The finite
population correction gives those stratum a contribution of 0 to the score variance, which 
ensures that our algorithm will terminate when we exhaust stratum.


## v 2019.04.23

Improved Summary Report class. Introduction of experiment running.

## v 2019.04.22

Improved Docstrings and Improved README

Additional Updates:
* Abstract class aegis.oracle.oracle.Oracle provided as a way to extend for other oracles
* Some key Mathematical formulas included as Images from rendered LaTeX
* Quickstart included in README



## v 2019.04.18

Refactoring complete. First prototype implementation working!

Additional updates:
* uses method names of score_variance and variance to distinguish whether we are estimating the
variance or the square of the standard error of the score
* updated computations of score_variance and variance to match stratified sampling formulas
* Improved documentation strings
* Controller is implemented
* Simple test cases add to test mathematics and execution of controller.

## v 2019.04.10

Refactoring still in progress. However, some updates:
* System Class objects now designed
* Three Samplers implemented (not yet tested): uniform, proportional, and optional
* subclassing rather than functions used for stratification strategies
* Strata subclass StrataFirstSystem 
* data_processor.extract_files_from_directory() now has an optional argument system_ordering, allowing users to specify which order the systems should appear. Very useful to configure which system is the first system for the
StrataFirstSystem class

## v 2019.03.22

First refactored version, refactoring in progress. Some changes:
* Metric classes used to specify metrics, with AccuracyMetric implemented
* Oracle class now provides samples. An implementation with a known key has been implemented.