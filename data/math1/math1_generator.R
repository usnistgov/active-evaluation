set.seed(418)
math_1_root_dir <- "/Users/pcf/git_repositories/dse/active-evaluation/data/test"
sys_sd_1 = 0.3
sys_sd_2 = 0.7
sys_sd_3 = 0.7
# Common Threshold DF
threshold_df <- data.frame(c(0.5))
names(threshold_df) <- "accuracy"

# # Construct first dataset: prior 0.5, 10000 trials
suffix = "p5_10000t"
dataset_dir <- normalizePath(file.path(math_1_root_dir, paste0("math1_", suffix)),
                             mustWork = FALSE)
# Make directory if it does not exist
if (!dir.exists(dataset_dir)) {
  dir.create(dataset_dir, recursive = TRUE)
}
num_trials = 10000
prior = 0.5
# Key is deterministically computed
key <- c(rep(0, (1 - prior)*num_trials), rep(1, prior*num_trials))
system_1 <- key + rnorm(num_trials, mean = 0, sd = sys_sd_1)
system_2 <- key + rnorm(num_trials, mean = 0, sd = sys_sd_2)
system_3 <- (1 - key) + rnorm(num_trials, mean = 0, sd = sys_sd_3)
test_df_1 <- data.frame(system_1, system_2, system_3, key)
test_df_1$trial_id <- as.numeric(rownames(test_df_1))
out_key_df <- test_df_1[, c("trial_id", "key")]
sys_1_df <- test_df_1[, c("trial_id", "system_1")]
sys_2_df <- test_df_1[, c("trial_id", "system_2")]
sys_3_df <- test_df_1[, c("trial_id", "system_3")]
names(sys_1_df) <- c("trial_id", "score")
names(sys_2_df) <- c("trial_id", "score")
names(sys_3_df) <- c("trial_id", "score")
key_fpath <- normalizePath(file.path(dataset_dir, "key.csv"), mustWork = FALSE)
s1_fpath <- normalizePath(file.path(dataset_dir, "s1_outputs.csv"), mustWork = FALSE)
s2_fpath <- normalizePath(file.path(dataset_dir, "s2_outputs.csv"), mustWork = FALSE)
s3_fpath <- normalizePath(file.path(dataset_dir, "s3_outputs.csv"), mustWork = FALSE)
ts1_fpath <- normalizePath(file.path(dataset_dir, "s1_thresholds.csv"), mustWork = FALSE)
ts2_fpath <- normalizePath(file.path(dataset_dir, "s2_thresholds.csv"), mustWork = FALSE)
ts3_fpath <- normalizePath(file.path(dataset_dir, "s3_thresholds.csv"), mustWork = FALSE)
write.csv(out_key_df, key_fpath, row.names = FALSE)
write.csv(sys_1_df, s1_fpath, row.names = FALSE)
write.csv(sys_2_df, s2_fpath, row.names = FALSE)
write.csv(sys_3_df, s3_fpath, row.names = FALSE)
# All systems have the same threshold
write.csv(threshold_df, ts1_fpath, row.names = FALSE)
write.csv(threshold_df, ts2_fpath, row.names = FALSE)
write.csv(threshold_df, ts3_fpath, row.names = FALSE)

# # Construct second dataset: prior 0.5, 50000 trials
suffix = "p5_50000t"
dataset_dir <- normalizePath(file.path(math_1_root_dir, paste0("math1_", suffix)),
                             mustWork = FALSE)
# Make directory if it does not exist
if (!dir.exists(dataset_dir)) {
  dir.create(dataset_dir, recursive = TRUE)
}
num_trials = 50000
prior = 0.5
key <- c(rep(0, (1 - prior)*num_trials), rep(1, prior*num_trials))
system_1 <- key + rnorm(num_trials, mean = 0, sd = sys_sd_1)
system_2 <- key + rnorm(num_trials, mean = 0, sd = sys_sd_2)
system_3 <- (1 - key) + rnorm(num_trials, mean = 0, sd = sys_sd_3)
test_df_1 <- data.frame(system_1, system_2, system_3, key)
test_df_1$trial_id <- as.numeric(rownames(test_df_1))
out_key_df <- test_df_1[, c("trial_id", "key")]
sys_1_df <- test_df_1[, c("trial_id", "system_1")]
sys_2_df <- test_df_1[, c("trial_id", "system_2")]
sys_3_df <- test_df_1[, c("trial_id", "system_3")]
names(sys_1_df) <- c("trial_id", "score")
names(sys_2_df) <- c("trial_id", "score")
names(sys_3_df) <- c("trial_id", "score")
key_fpath <- normalizePath(file.path(dataset_dir, "key.csv"), mustWork = FALSE)
s1_fpath <- normalizePath(file.path(dataset_dir, "s1_outputs.csv"), mustWork = FALSE)
s2_fpath <- normalizePath(file.path(dataset_dir, "s2_outputs.csv"), mustWork = FALSE)
s3_fpath <- normalizePath(file.path(dataset_dir, "s3_outputs.csv"), mustWork = FALSE)
ts1_fpath <- normalizePath(file.path(dataset_dir, "s1_thresholds.csv"), mustWork = FALSE)
ts2_fpath <- normalizePath(file.path(dataset_dir, "s2_thresholds.csv"), mustWork = FALSE)
ts3_fpath <- normalizePath(file.path(dataset_dir, "s3_thresholds.csv"), mustWork = FALSE)
write.csv(out_key_df, key_fpath, row.names = FALSE)
write.csv(sys_1_df, s1_fpath, row.names = FALSE)
write.csv(sys_2_df, s2_fpath, row.names = FALSE)
write.csv(sys_3_df, s3_fpath, row.names = FALSE)
# All systems have the same threshold
write.csv(threshold_df, ts1_fpath, row.names = FALSE)
write.csv(threshold_df, ts2_fpath, row.names = FALSE)
write.csv(threshold_df, ts3_fpath, row.names = FALSE)

# # Construct third dataset: prior 0.8, 10000 trials
suffix = "p8_10000t"
dataset_dir <- normalizePath(file.path(math_1_root_dir, paste0("math1_", suffix)),
                             mustWork = FALSE)
# Make directory if it does not exist
if (!dir.exists(dataset_dir)) {
  dir.create(dataset_dir, recursive = TRUE)
}
num_trials = 10000
prior = 0.8
# Key is deterministically computed
# Using manual values due to a bug in rep()
key <- c(rep(0, 2000), rep(1, prior*num_trials))
system_1 <- key + rnorm(num_trials, mean = 0, sd = sys_sd_1)
system_2 <- key + rnorm(num_trials, mean = 0, sd = sys_sd_2)
system_3 <- (1 - key) + rnorm(num_trials, mean = 0, sd = sys_sd_3)
test_df_1 <- data.frame(system_1, system_2, system_3, key)
test_df_1$trial_id <- as.numeric(rownames(test_df_1))
out_key_df <- test_df_1[, c("trial_id", "key")]
sys_1_df <- test_df_1[, c("trial_id", "system_1")]
sys_2_df <- test_df_1[, c("trial_id", "system_2")]
sys_3_df <- test_df_1[, c("trial_id", "system_3")]
names(sys_1_df) <- c("trial_id", "score")
names(sys_2_df) <- c("trial_id", "score")
names(sys_3_df) <- c("trial_id", "score")
key_fpath <- normalizePath(file.path(dataset_dir, "key.csv"), mustWork = FALSE)
s1_fpath <- normalizePath(file.path(dataset_dir, "s1_outputs.csv"), mustWork = FALSE)
s2_fpath <- normalizePath(file.path(dataset_dir, "s2_outputs.csv"), mustWork = FALSE)
s3_fpath <- normalizePath(file.path(dataset_dir, "s3_outputs.csv"), mustWork = FALSE)
ts1_fpath <- normalizePath(file.path(dataset_dir, "s1_thresholds.csv"), mustWork = FALSE)
ts2_fpath <- normalizePath(file.path(dataset_dir, "s2_thresholds.csv"), mustWork = FALSE)
ts3_fpath <- normalizePath(file.path(dataset_dir, "s3_thresholds.csv"), mustWork = FALSE)
write.csv(out_key_df, key_fpath, row.names = FALSE)
write.csv(sys_1_df, s1_fpath, row.names = FALSE)
write.csv(sys_2_df, s2_fpath, row.names = FALSE)
write.csv(sys_3_df, s3_fpath, row.names = FALSE)
# All systems have the same threshold
write.csv(threshold_df, ts1_fpath, row.names = FALSE)
write.csv(threshold_df, ts2_fpath, row.names = FALSE)
write.csv(threshold_df, ts3_fpath, row.names = FALSE)

# # Construct fourth dataset: prior 0.8, 50000 trials
suffix = "p8_50000t"
dataset_dir <- normalizePath(file.path(math_1_root_dir, paste0("math1_", suffix)),
                             mustWork = FALSE)
# Make directory if it does not exist
if (!dir.exists(dataset_dir)) {
  dir.create(dataset_dir, recursive = TRUE)
}
num_trials = 50000
prior = 0.8
# Using manual values due to a bug in rep()
key <- c(rep(0, 10000), rep(1, prior*num_trials))
system_1 <- key + rnorm(num_trials, mean = 0, sd = sys_sd_1)
system_2 <- key + rnorm(num_trials, mean = 0, sd = sys_sd_2)
system_3 <- (1 - key) + rnorm(num_trials, mean = 0, sd = sys_sd_3)
test_df_1 <- data.frame(system_1, system_2, system_3, key)
test_df_1$trial_id <- as.numeric(rownames(test_df_1))
out_key_df <- test_df_1[, c("trial_id", "key")]
sys_1_df <- test_df_1[, c("trial_id", "system_1")]
sys_2_df <- test_df_1[, c("trial_id", "system_2")]
sys_3_df <- test_df_1[, c("trial_id", "system_3")]
names(sys_1_df) <- c("trial_id", "score")
names(sys_2_df) <- c("trial_id", "score")
names(sys_3_df) <- c("trial_id", "score")
key_fpath <- normalizePath(file.path(dataset_dir, "key.csv"), mustWork = FALSE)
s1_fpath <- normalizePath(file.path(dataset_dir, "s1_outputs.csv"), mustWork = FALSE)
s2_fpath <- normalizePath(file.path(dataset_dir, "s2_outputs.csv"), mustWork = FALSE)
s3_fpath <- normalizePath(file.path(dataset_dir, "s3_outputs.csv"), mustWork = FALSE)
ts1_fpath <- normalizePath(file.path(dataset_dir, "s1_thresholds.csv"), mustWork = FALSE)
ts2_fpath <- normalizePath(file.path(dataset_dir, "s2_thresholds.csv"), mustWork = FALSE)
ts3_fpath <- normalizePath(file.path(dataset_dir, "s3_thresholds.csv"), mustWork = FALSE)
write.csv(out_key_df, key_fpath, row.names = FALSE)
write.csv(sys_1_df, s1_fpath, row.names = FALSE)
write.csv(sys_2_df, s2_fpath, row.names = FALSE)
write.csv(sys_3_df, s3_fpath, row.names = FALSE)
# All systems have the same threshold
write.csv(threshold_df, ts1_fpath, row.names = FALSE)
write.csv(threshold_df, ts2_fpath, row.names = FALSE)
write.csv(threshold_df, ts3_fpath, row.names = FALSE)