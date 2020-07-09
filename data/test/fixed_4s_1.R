data_dir <- "/Users/pcf/git_repositories/dse/active-evaluation/data/test/fixed_4s_1"

num_trials = 20000
# Get 50% of the values as -1, 50% as -1, determinized
base_df <- as.data.frame(seq(1,20000,by = 1))
names(base_df) <- "trial_id"
base_df$perc <- (base_df$index/num_trials)*100
key <- c(rep(0, 0.7*num_trials), rep(1, 0.3*num_trials))
base_df$key <- key
# Intended for proportional stratification, 4 stratum
s1_vec <- c(rep(0.1, 0.25*num_trials), rep(0.6, 0.25*num_trials), rep(0.35, 0.15*num_trials),
              rep(0.85, 0.05*num_trials), rep(0.35, 0.10*num_trials), rep(0.85, 0.20*num_trials) ) 
base_df$s1 <- s1_vec
table(base_df$s1, base_df$key)
s2_vec <- c(rep(0.1, 0.25*num_trials), rep(0.15, 0.25*num_trials), rep(0.2, 0.15*num_trials),
            rep(0.9, 0.05*num_trials), rep(0.2, 0.10*num_trials), rep(0.9, 0.20*num_trials) ) 
base_df$s2 <- s2_vec
table(base_df$s2, base_df$key)
# Intended for either stratification, 4 stratum
srh_vec <- c(rep(0.2, 0.175*num_trials), rep(0.4, 0.175*num_trials), rep(0.6, 0.175*num_trials),
             rep(0.8, 0.175*num_trials), rep(0.2, 0.075*num_trials), rep(0.4, 0.075*num_trials),
             rep(0.6, 0.075*num_trials), rep(0.8, 0.075*num_trials)) 
base_df$srh <- srh_vec
table(base_df$srh, base_df$key)
skey_vec <- c(rep(0.1, 0.25*num_trials), rep(0.35, 0.25*num_trials), rep(0.45, 0.2*num_trials),
              rep(0.55,0.05*num_trials), rep(0.85, 0.25*num_trials))
base_df$skey <- skey_vec
table(base_df$skey, base_df$key)

out_key_df <- base_df[,c("trial_id", "key")]
s1_df <- base_df[,c("trial_id", "s1")]
names(s1_df) <- c("trial_id", "score")
s2_df <- base_df[,c("trial_id", "s2")]
names(s2_df) <- c("trial_id", "score")
srh_df <- base_df[,c("trial_id", "srh")]
names(srh_df) <- c("trial_id", "score")
skey_df <- base_df[,c("trial_id", "skey")]
names(skey_df) <- c("trial_id", "score")
write.csv(out_key_df, normalizePath(file.path(data_dir, "key.csv"), mustWork = FALSE), row.names = FALSE)
write.csv(s1_df, normalizePath(file.path(data_dir, "s1_outputs.csv"), mustWork = FALSE), row.names = FALSE)
write.csv(s2_df, normalizePath(file.path(data_dir, "s2_outputs.csv"), mustWork = FALSE), row.names = FALSE)
write.csv(srh_df, normalizePath(file.path(data_dir, "srh_outputs.csv"), mustWork = FALSE), row.names = FALSE)
write.csv(skey_df, normalizePath(file.path(data_dir, "skey_outputs.csv"), mustWork = FALSE), row.names = FALSE)

# System 1 computations of score and score_variance per stratum without finite population correction
# > table(base_df$s1, base_df$key)
# 
#         0    1
# 0.1  5000    0
# 0.35 3000 2000
# 0.6  5000    0
# 0.85 1000 4000
s1_score = 0.25*1 + 0.25*0.6 + 0.25*0 + 0.25*0.8 = 0.6
s1_val_stratum = c(0.1, 0.35, 0.6, 0.85)
s1_score_stratum = c(1,0.6,0,0.8)
s1_var_stratum = c(0, 0.24, 0, 0.16)
# If we have 10 samples per stratum without FPC
s1_10_score_var_stratum = c(0, 0.024, 0, 0.016)
s1_10_score_var = (0.25^2)*0 + (0.25^2)*0.024 + (0.25^2)*0 + (0.25^2)*0.016
s1_10_conf_value = qnorm(1 - 0.05/2)*sqrt(s1_10_score_var)

# If we have 10 samples per stratum without FPC but add in +2 each tp/fp for agresti
# score unchanged,
s1_10_score_agr = c(0.8571429, 0.5714286, 0.1428571, 0.7142857)
s1_10_var_agr = c(0.1224489, 0.244898, 0.1224489, 0.2040816)
s1_10_score_var_str_agr = c(0.00874635, 0.01749271, 0.00874635, 0.01457726)
s1_10_score_var_agr = (0.25^2)*0.00874635 + (0.25^2)*0.01749271 + 
  (0.25^2)*0.00874635 + (0.25^2)*0.01457726
s1_10_conf_value_agr = qnorm(1 - 0.05/2)*sqrt(s1_10_score_var_agr)

# If we have 25 samples per stratum without FPC
s1_25_score_var_stratum = c(0, 0.0096, 0, 0.0064)
s1_25_conf_value
