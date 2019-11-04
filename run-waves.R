############################################################################################
################################# Run waves model ##########################################
############################################################################################

# setwd() # set your working directory to the folder containing this script
# Then make sure that folder contains the scripts and data folders

rm(list=ls())

# Load the dataset - pre-declustered excesses above the 99th threshold in this example:
load("data/waves.RData")

# Combining these into a list for later:
my_data_waves <- list(final_data, covar, mu)

# final_data now has 334 elements, one for each point on the grid

# Find some suitable starting values:
library(ismev)

list1 <- list()
for (i in 1:length(final_data)) {
  list1[[i]] <- gpd.fit(final_data[[i]], threshold = mu[i], npy = 24 * 365.25, show = FALSE)
}

start_scale <- unlist(lapply(list1, function(x) x$mle[1]))
start_shape <- unlist(lapply(list1, function(x) x$mle[2]))

########################### Setting things up: ###########################
# Adjust priors, steps, starting point etc. as required:

# The length of the dataset:
len <- length(final_data)

# Find the number of covariates:
# (lat and lon in this example)
no.cov <- ncol(covar)

# Set starting values for the MCMC algorithm (change these as desired):
start <- list('lscale'              = log(start_scale),
              'alpha_scale'         = c(rep(0, no.cov)),
              'beta_scale'          = matrix(data=c(1, 0.5, 0.5, 1), nrow=2),
              'sigsq_scale'         = 1,
              'tausq_scale'         = 0.1,
              'shape'               = start_shape,
              'alpha_shape'         = c(rep(0, no.cov)),
              'beta_shape'          = matrix(data=c(1, 0, 0, 1), nrow=2),
              'sigsq_shape'         = 1,
              'tausq_shape'         = 0.1)

# Set prior values for the MCMC algorithm (change these as desired):
prior <- list('alpha_scale_hyper_mean'  = c(rep(0, no.cov)),
              'alpha_scale_hyper_sd'    = diag(rep(50, no.cov)),
              'beta_scale_prior'        = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000),
              'alpha_shape_hyper_mean'  = c(rep(0, no.cov)),
              'alpha_shape_hyper_sd'    = diag(rep(1, no.cov)),
              'beta_shape_prior'        = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000),
              'shape_hyper_mean'        = 0,
              'shape_hyper_sd'          = 5,
              'sigsq_scale_mean'        = 1,
              'sigsq_scale_sd'          = 0.5,
              'tausq_scale_mean'        = 0.1,
              'tausq_scale_sd'          = 0.05,
              'sigsq_shape_mean'        = 1,
              'sigsq_shape_sd'          = 0.5,
              'tausq_shape_mean'        = 0.1,
              'tausq_shape_sd'          = 0.05)

# Set the step values for the MCMC algorithm (change these as desired):
step <- list('lscale_step'      = 0.04,
             'alpha_scale_step' = c(0.2, 0.2, 0.2, 0, 0),
             'sigsq_scale_step' = 1,
             'tausq_scale_step' = 0.6,
             'shape_step'       = 0.002,
             'alpha_shape_step' = c(0.1, 0.08, 0.08, 0, 0),
             'sigsq_shape_step' = 1,
             'tausq_shape_step' = 0.5)

# Loading packages and the MCMC Rcpp script:
library(Rcpp)
library(RcppArmadillo)
sourceCpp("scripts/mcmc_gpd.cpp") # A warning or two about unused variables - can be ignored

########################### Running the MCMC code: ###########################
# Running the MCMC (adjust the iterations, burnin and thinning value as desired):
out <- mcmcC(final_data, mu, start,
             iterations=1e2, covar, covar, step, prior,
             burnin=1e1, nth=1e1)

# Write the data and the chain to a file:
save(my_data_waves, out, file="final_gpd_chains.RData")

# break() here, so script can be sourced using a bash script etc.
break()

############################################################################################
################################# Post-run analysis ########################################
############################################################################################
load("final_gpd_chains.RData")

# Get the posterior medians:
scale <- apply(out$scale, 2, median)
shape <- apply(out$shape, 2, median)

# Lower and upper bounds:
scale_low <- apply(out$scale, 2, function(x) quantile(x, 0.025))
scale_high <- apply(out$scale, 2, function(x) quantile(x, 0.975))

shape_low <- apply(out$shape, 2, function(x) quantile(x, 0.025))
shape_high <- apply(out$shape, 2, function(x) quantile(x, 0.975))

# Save these (and the location info) for plotting using Python etc.:
output <- cbind(x=my_data_waves[[2]][,2], y=my_data_waves[[2]][,3],
                scale, scale_low, scale_high,
                shape, shape_low, shape_high)
write.csv(output, file="0.99_gpd_data_to_plot.csv")

##############################################
# A quick check of the hyperparameter chains:

# alpha scale:
for (i in 1:ncol(out$alpha_scale)) {
  plot(seq(1:length(out$alpha_scale[,i])), out$alpha_scale[,i], type="l")
}

# beta scale (the top corner, off-diagonal, and bottom corner):
table(out$beta_scale[,1])
table(out$beta_scale[,2])
table(out$beta_scale[,4])

# sigma scale:
plot(seq(1:length(out$sigsq_scale)), out[[4]], type="l")

# tausq scale:
plot(seq(1:length(out$tausq_scale)), out$sigsq_scale, type="l")

# alpha shape:
for (i in 1:ncol(out$alpha_shape)) {
  plot(seq(1:length(out$alpha_shape[,i])), out$alpha_shape[,i], type="l")
}

# beta shape (the top corner, off-diagonal, and bottom corner):
table(out$beta_shape[,1])
table(out$beta_shape[,2])
table(out$beta_shape[,4])

# sigma shape:
plot(seq(1:length(out$sigsq_shape)), out$sigsq_shape, type="l")

# tausq shape:
plot(seq(1:length(out$tausq_shape)), out$tausq_shape, type="l")

########################################################################
############## Producing a nice plot of a chain: #######################
########################################################################

df2 <- as.data.frame(cbind(x=seq(1:length(out[[2]][,1])),
                           y=out[[2]][,1]))
library(ggplot2)
pp <- ggplot(df2, aes(x, y)) +
  geom_line() +
  xlab("Index") +
  ylab(expression(paste(alpha[1]))) +
  ggtitle(expression(paste("MCMC chain of scale hyperparameter ", alpha[1])))
pp
ggsave(filename="alpha_sample_chain.pdf", plot=pp)

########################################################################
############### Next, to run the binomial model: #######################
########################################################################

### To run this model:
### I need all of these:
### List mcmc_binom_C(vec counts, vec total_obs, List start, int iterations, mat covar1,
### List step, List prior, int burnin, int nth)

# Source the script:
Rcpp::sourceCpp('scripts/mcmc_binom.cpp')

# Get the data ready:
counts <- unlist(lapply(my_data_waves[[1]], length))
total_obs <- rep(34 * 24 * 365.25, length(counts))

len <- length(total_obs)

# Again, just lat and lon in this example:
covar <- my_data_waves[[2]][,1:ncol(my_data_waves[[2]])]
no.cov <- dim(covar)[2]

# Transform by the logit to get the starting values:
zeta <- log(counts/total_obs) - log(1 - counts/total_obs)

########################### Setting things up: ###########################
# Adjust priors, steps, starting point etc. as required:

# Set starting values for the MCMC algorithm (change these as desired):
start <- list('zeta'               = zeta,
              'alpha_zeta'         = c(rep(0, no.cov)),
              'beta_zeta'          = matrix(data=c(1, 0.5, 0.5, 1), nrow=2),
              'sigsq_zeta'         = 1,
              'tausq_zeta'         = 0.1)

# Set prior values for the MCMC algorithm (change these as desired):
prior <- list('alpha_zeta_hyper_mean'  = c(rep(0, no.cov)),
              'alpha_zeta_hyper_sd'    = diag(rep(5, no.cov)),
              'beta_zeta_prior'        = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000))

# Set step values for the MCMC algorithm (change these as desired):
step <- list('zeta_step'       = 0.008,
             'alpha_zeta_step' = c(0.2, 0.06, 0.05, 0.001, 0.07),
             'sigsq_zeta_step' = 0.5,
             'tausq_zeta_step' = 0.4)

########################### Running the MCMC code: ###########################
# Run the big MCMC code (adjust iterations, burnin, thinning (nth) as required):
out2 <- mcmc_binom_C(counts, total_obs, start,
                     iterations = 1e2,
                     covar, step, prior,
                     burnin = 1e1, nth = 1e1)

# Write the data and the chain to a file:
save(my_data_waves, out2, file="final_binom_chains.RData")

# Pick out the posterior zeta (back-transforming):
prob <- exp(out2$zeta) / ( 1 + exp(out2$zeta) )

# Calculate posterior values if wanted, ready to plot and use etc.:
prob_med <- apply(prob, 2, median)
prob_low <- apply(prob, 2, function(x) quantile(x, 0.025))
prob_high <- apply(prob, 2, function(x) quantile(x, 0.975))
prob_stdev <- apply(prob, 2, function(x) sqrt(var(x)))

# Save these (and the location info) for plotting using Python etc.:
output <- cbind(x=my_data_waves[[2]][,2], y=my_data_waves[[2]][,3],
                prob_med, prob_low,
                prob_high, prob_stdev)
write.csv(output, file="binom_data_to_plot.csv")

############################################################################################
############################## Run waves model: End ########################################
############################################################################################
