set.seed(5)
num_trials = 10000
sys_sd = 0.3
key <- sample.int(2,size=num_trials, replace=TRUE) - 1
system_1 <- key + rnorm(10000, mean = 0, sd = sys_sd)
system_2 <- rep(0,num_trials)
test_df_1 <- data.frame(system_1,system_2,key)
test_df_1$trial_id <- as.numeric(rownames(test_df_1))
out_key_df <- test_df_1[,c("trial_id", "key")]
sys_1_df <- test_df_1[,c("trial_id", "system_1")]
sys_2_df <- test_df_1[,c("trial_id", "system_2")]
names(sys_1_df) <- c("trial_id", "score")
names(sys_2_df) <- c("trial_id", "score")
write.csv(out_key_df, "~/Desktop/sae_test_1_key.csv",row.names=FALSE)
write.csv(sys_1_df, "~/Desktop/sae_test_1_sys_1.csv",row.names=FALSE)
write.csv(sys_2_df, "~/Desktop/sae_test_1_sys_2.csv",row.names=FALSE)