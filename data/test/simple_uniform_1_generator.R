set.seed(57)
num_trials = 10002
# Get 50% of the values as -1, 50% as -1, determinized
key <- c(rep(-1, num_trials/2), rep(1, num_trials/2))
system_1 <- c(seq(-2, 0, by = 0.0004), seq(-0.5, 2.5, by = 0.0006)) 
test_df_1 <- data.frame(system_1, key)
test_df_1$trial_id <- as.numeric(rownames(test_df_1))
out_key_df <- test_df_1[,c("trial_id", "key")]
sys_1_df <- test_df_1[,c("trial_id", "system_1")]
# Every 100th trial
init_df <- out_key_df[seq(1, nrow(out_key_df), 100), ]
names(sys_1_df) <- c("trial_id", "score")
write.csv(out_key_df, "~/Desktop/key.csv", row.names = FALSE)
write.csv(sys_1_df, "~/Desktop/s_uniform_outputs.csv", row.names = FALSE)
write.csv(init_df, "~/Desktop/init.csv", row.names = FALSE)
