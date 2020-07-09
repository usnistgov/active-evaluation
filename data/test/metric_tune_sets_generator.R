set.seed(1712)
# Set 1, Generate 10000 Trials, give three different initial data frames for metric computation
# Systems for testng
num_trials = 10000
# 2 is high value, 1 is low value, but only 20% high value
key <- sample.int(2,size = num_trials, replace = TRUE, prob = c(0.8, 0.2))
system_n1 <- key + rnorm(10000, mean = 0, sd = 0.3)
system_n2 <- (3 - key) + rnorm(10000, mean = 0, sd = 0.7)
system_ni1 <- rep(1, num_trials)
system_ni2 <- rep(2, num_trials)
test_df_1 <- data.frame(system_n1, system_n2, system_ni1, system_ni2, key)
test_df_1$trial_id <- as.numeric(rownames(test_df_1))
out_key_df <- test_df_1[, c("trial_id", "key")]
sys_n1_df <- test_df_1[, c("trial_id", "system_n1")]
sys_n2_df <- test_df_1[, c("trial_id", "system_n2")]
sys_ni1_df <- test_df_1[, c("trial_id", "system_ni1")]
sys_ni2_df <- test_df_1[, c("trial_id", "system_ni2")]
names(sys_n1_df) <- c("trial_id", "score")
names(sys_n2_df) <- c("trial_id", "score")
names(sys_ni1_df) <- c("trial_id", "score")
names(sys_ni2_df) <- c("trial_id", "score")
write.csv(out_key_df, "~/Desktop/key.csv", row.names = FALSE)
write.csv(sys_n1_df, "~/Desktop/n1_outputs.csv", row.names = FALSE)
write.csv(sys_n2_df, "~/Desktop/n2_outputs.csv", row.names = FALSE)
write.csv(sys_ni1_df, "~/Desktop/ni1_outputs.csv", row.names = FALSE)
write.csv(sys_ni2_df, "~/Desktop/ni2_outputs.csv", row.names = FALSE)
# Compute Fixed initial sets:
test_df_n1tn <- test_df_1[(test_df_1$system_n1 < 1.5) & (test_df_1$key == 1), ]
test_df_n1fn <- test_df_1[(test_df_1$system_n1 < 1.5) & (test_df_1$key == 2), ]
test_df_n1tp <- test_df_1[(test_df_1$system_n1 > 1.5) & (test_df_1$key == 2), ]
test_df_n1fp <- test_df_1[(test_df_1$system_n1 > 1.5) & (test_df_1$key == 1), ]
# 10 samples: 2 tn, 2 tp, 3 fn, 3 fp
init_10f_df <- rbind(test_df_n1tn[1:2,], test_df_n1fn[1:3,], test_df_n1tp[1:2,],
                     test_df_n1fp[1:3,])
init_10_df <- init_10f_df[, c("trial_id", "key")]
# 20 samples: 5 tn, 5 tp, 5 fn, 5 fp
write.csv(init_10_df, "~/Desktop/init.csv", row.names = FALSE)
init_20f_df <- rbind(test_df_n1tn[1:5,], test_df_n1fn[1:5,], test_df_n1tp[1:5,],
                     test_df_n1fp[1:5,])
init_20_df <- init_20f_df[, c("trial_id", "key")]
write.csv(init_20_df, "~/Desktop/init.csv", row.names = FALSE)
# 100 initial samples: 25 tn, 25 tp, 25 fn, 25 fp
init_100f_df <- rbind(test_df_n1tn[1:25,], test_df_n1fn[1:25,], test_df_n1tp[1:25,],
                      test_df_n1fp[1:25,])
init_100_df <- init_100f_df[, c("trial_id", "key")]
write.csv(init_100_df, "~/Desktop/init.csv", row.names = FALSE)
# 853 initial samples: 250 tn, 250 tp, 250 fp, 103 fn (Every fn is chosen)
init_853f_df <- rbind(test_df_n1tn[1:250,], test_df_n1fn[1:103,], test_df_n1tp[1:250,],
                       test_df_n1fp[1:250,])
init_853_df <- init_853f_df[, c("trial_id", "key")]
write.csv(init_853_df, "~/Desktop/init.csv", row.names = FALSE)