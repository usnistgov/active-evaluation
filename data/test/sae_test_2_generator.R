set.seed(561)
num_trials = 10000
sys_sd_1 = 0.3
sys_sd_2 = 0.7
sys_sd_3 = 0.7
key <- sample.int(2,size=num_trials, replace=TRUE) - 1
system_1 <- key + rnorm(10000, mean = 0, sd = sys_sd_1)
system_2 <- key + rnorm(10000, mean = 0, sd = sys_sd_2)
system_3 <- (1 - key) + rnorm(10000, mean = 0, sd = sys_sd_3)
test_df_1 <- data.frame(system_1,system_2,system_3,key)
test_df_1$trial_id <- as.numeric(rownames(test_df_1))
out_key_df <- test_df_1[,c("trial_id", "key")]
sys_1_df <- test_df_1[,c("trial_id", "system_1")]
sys_2_df <- test_df_1[,c("trial_id", "system_2")]
sys_3_df <- test_df_1[,c("trial_id", "system_3")]
names(sys_1_df) <- c("trial_id", "score")
names(sys_2_df) <- c("trial_id", "score")
names(sys_3_df) <- c("trial_id", "score")
write.csv(out_key_df, "~/Desktop/sae_test_2_key.csv",row.names=FALSE)
write.csv(sys_1_df, "~/Desktop/sae_test_2_sys_1.csv",row.names=FALSE)
write.csv(sys_2_df, "~/Desktop/sae_test_2_sys_2.csv",row.names=FALSE)
write.csv(sys_3_df, "~/Desktop/sae_test_2_sys_3.csv",row.names=FALSE)