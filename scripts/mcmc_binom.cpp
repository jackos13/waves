//////// Cpp Binom Script - Version 3 ////////////
//////// This script is for the binomial parameter, zeta -
//////// It will contain the code for the MCMC process to target
//////// posterior distributions of zeta, given the data (counts, and total number of obs. at each point)
// ##################### Functions needed for model 2 - the Binmoial ###################
// ################################################################################

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calculating the distance list:
// This function takes an n x 2 matrix and returns a list. Each component is an n x 2 matrix showing the distance
// between its row and all the other rows. e.g. element 1 in the list has its first row as the distance between
// point 1 and point 1, the second row as the distance between point 1 and point 2 etc.
// (Could adapt this for three-dimensional grids if needed.)
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
List dist_mat(mat xy) {
  int n = xy.n_rows;
  List out(n);
  mat temp(n, 2);
  
  for (int i=0; i < n; ++i) {
    for (int j=0; j < n; ++j) {
      temp.row(j) = xy.row(i) - xy.row(j);
    }
    out[i] = temp;
  }
  
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calculating the var/covar matrix using the distance, beta, sigsq and tausq as inputs:
// This function calculates the big covar matrix Sigma needed for all the GPs -
// Sigma = sigsq * exp ( -d_ij * (beta^-1) * transpose(d_ij) ) + tausq * Identity
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
mat get_mat_sigma(List dist, mat beta, double sigsq, double tausq) {
  int n = dist.size();
  
  mat sig = zeros(n, n);
  mat tmpi(n, 2);
  mat invb = inv(beta);
  
  for (int i=0; i < n; ++i) {
    tmpi = as<mat>(dist[i]);
    for (int j=0; j <= i; ++j) {
      sig(i, j) = exp( - as_scalar( tmpi.row(j) * invb * tmpi.row(j).t() ) );
      if (i == j) {
        sig(i, j) = sig(i, j) + tausq;
      } else {
        sig(j, i) = sig(i, j);
      }   
    }
  }
  
  return sigsq * sig;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ################################################################################
// ############################# Data layer #######################################
// ################################################################################
// The full Binomial, works on a list:
// This function, on the log scale, takes in a vector of counts for each gridpoint, and a vector of total
// obs. corresponding to those (allowing for the fact that there may be different length time series
// at different grid points) and a vector of probabilities at each point too.
// Note that the factorial component is not needed, since it appears identically in the
// denominator and the numerator, and thus will cancel each time  
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double dbinomC(vec counts, vec total_obs, vec zeta) {
  
  // Converting back into a vector of probs. between 0 and 1
  vec prob = exp(zeta) / (1 + exp(zeta));
  
  // This is the (1 - p_i) piece:
  vec inv_prob = 1 - prob;
  
  // This is the k_i * log(p_i) piece:
  vec a = counts % log(prob);
  
  // This is the (n_i - k_i) * log(1 - p_i) piece:
  vec b = (total_obs - counts) % log(inv_prob);
  
  // Finally to sum the two vectors, and sum all entries:
  double out = sum(a + b);
  
  return out;
}

// ################################################################################
// ############################# The latent process layer #########################
// ################################################################################
// The MVN in log form, for proportions: mvnC
// This gives the same value os dmvnorm(x, mu, sigma, log=T)
// when the (-k/2 * log(2 * pi)) factor is included at the end. This will be above and below, and will cancel
#define _USE_MATH_DEFINES
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// Now for the MVN:
// [[Rcpp::export]]
double mvnC(vec x, vec mu, mat sigma) {
  
  double x1;
  double sign;
  log_det(x1, sign, sigma); // This gets the log of the determinant of the covar matrix sigma.
  
  vec p = x - mu;
  double c = arma::as_scalar(p.t() * solve(sigma, p));
  double d = 0.5 * c;
  return (-0.5) * x1 - d;
}


// ################################################################################
// The MVN in log form and using the precision, for proportions: mvnC2 
// Here, the precision is used - where it's calculated as the inverse of sigma directly within
// the MCMC code. This formula agrees with the one above (luckily!)
#define _USE_MATH_DEFINES
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// Now for the MVN:
// [[Rcpp::export]]
double mvnC2(vec x, vec mu, mat invsigma, double dett) {
  
  vec p = x - mu;
  double c = arma::as_scalar(p.t() * invsigma * p);
  double d = 0.5 * c;
  
  return (-0.5) * dett - d;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Using mvnC for the GP distribution:
// This is the final component needed for layer 2 - the mvn applied from the previous formulae, but using
// the covariates too in order to calculate mu:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double gp_mvnC(vec zeta, mat invsigma, vec alpha, mat covar, double dett) {
  
  vec mean = covar * alpha;
  
  double out = mvnC2(zeta, mean, invsigma, dett);
  return out;
}

// #############################################################################################################
// ############################# Priors on hyperparameters (layer 3) ###########################################
// #############################################################################################################
// Evaluating the normal density component-wise:
// This takes a vector x, a mean vector mu, and a standard deviation vector s
// All must be the same length
// It then calculates pointwise densities for each element of the vector,
// gets the log, then sums the result to give a single number.
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double dnormvecC(vec x, vec m, vec s) {
  int n = x.size();
  
  vec out(n);
  
  double c = 1/sqrt(2*M_PI);
  
  vec f = (x - m) / s;
  vec ff = f % f;
  out = c* (1/s) % exp ( - 0.5 * ff);
  vec ret = log(out);
  
  return sum(ret);
}

// ################################################################################
// Evaluating the normal density for scalars:
// This straightforward function evaluates the normal density for a single scalar, given a mean
// and an sd. It then takes the log of this, and returns the value.
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double dnormscalarC(double x, double m, double s) {
  
  double out;
  
  double c = 1/sqrt(2*M_PI);
  
  double f = pow((x - m) / s, 2);
  out = c * (1/s) * exp ( - 0.5 * f);
  
  return log(out);
}

// ################################################################################
// A function to test for positive definitness:
// This function gets the det(big matrix) and saves it. It then sheds the final row and column and repeats the process
// At the end, a check is done that all entries are positive, as this is a sufficient requirement for a
// matrix to be positive definite. 
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
bool pos_def_testC(mat m) {
  int n = m.n_rows;
  vec v(n);
  
  for (int j=0; j<n; ++j) {
    v(j) = det(m); //round(det(m)); This was causing trouble at some stage
    m.shed_row(n-j-1);
    m.shed_col(n-j-1);
  }
  
  bool test = all(v>0);  
  return test;
}

// ###############################################################################################
// ############################################ Posteriors #######################################
// ###############################################################################################

// In this section, the necessary components for each posterior density are calculated
// These are in log form (or more precisely, the functions which they call are)
// Then within the MCMC code, the relevant fractions are evaluated as exp ( log (top) - log (bottom) )
// in order to get the ratio needed

// ##########################################################################################
// Evaluating the posterior conditional of zeta:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_zetaC(vec counts, vec total_obs, vec zeta, vec alpha, mat covar,
                  mat invsigma, int l, double dett_zeta) {
  double a = dbinomC(counts, total_obs, zeta);
  double b = gp_mvnC(zeta, invsigma, alpha, covar, dett_zeta);
  return a + b;
}

// ################################################################################
// Evaluating the posterior conditional of the alpha_zeta parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_alpha_zetaC(vec alpha_zeta, vec zeta, mat covar, mat invsigma,
                        vec alpha_zeta_hyper_mean, mat alpha_zeta_hyper_sd, double dett_zeta) {
  double a = gp_mvnC(zeta, invsigma, alpha_zeta, covar, dett_zeta); // Layer2
  double b = mvnC(alpha_zeta, alpha_zeta_hyper_mean, alpha_zeta_hyper_sd) ; //Layer3
  return a + b;
}  

// ################################################################################
// Evaluating the posterior conditional of the beta_zeta parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_beta_zetaC(mat invsigma, mat beta_zeta, vec zeta, vec alpha_zeta, mat covar, double dett_zeta) {
  double a = gp_mvnC(zeta, invsigma, alpha_zeta, covar, dett_zeta); // Layer2
  //double b =  ; //Layer3 - blank for now
  return a;
}  

// ################################################################################
// Evaluating the posterior conditional of the sigsq_zeta parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_sigsq_zetaC(mat invsigma, vec zeta, vec alpha, mat covar, double sigsq, double dett_zeta) {
  double a = gp_mvnC(zeta, invsigma, alpha, covar, dett_zeta); // Layer2
  //double b =  ; //Layer3 - blank for now
  return a;
}

// ################################################################################
// Evaluating the posterior conditional of the tausq_zeta parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_tausq_zetaC(mat invsigma, vec zeta, vec alpha, mat covar, double tausq, double dett_zeta) {
  double a = gp_mvnC(zeta, invsigma, alpha, covar, dett_zeta); // Layer2
  //double b =  ; //Layer3 - blank for now
  return a;
}

// ################################################################################

// Alternatives to the GP here, for comparison: Assuming that zeta is a constant value:

// Evaluating the full posterior conditional of zeta, where it is assumed constant:
// (Can include a prior on this too, by uncommenting below)
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_zetaC2(vec counts, vec total_obs, vec zeta) {
  double a = dbinomC(counts, total_obs, zeta);
  //  double b = dnormscalarC(shape, prior_mean, prior_sd);
  return a;
}

// ################################################################################
// ############################# Random updates ###################################
// ################################################################################
// This function takes a mean and an SD and returns a random value drawn from that Normal:

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double rnormscalarC(double m, double s) {
  vec x = rnorm(1) * s + m;
  return as_scalar(x);
}

// ################################################################################
// This code is to keep track of the acceptance rates within the MCMC process
// It compares vectors entry by entry
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp ;

// [[Rcpp::export]]
LogicalVector keep_acc(vec x, vec y) {
  
  LogicalVector r(x.size());
  for( int i=0; i<x.size(); i++){
    r[i] = (x[i] == y[i]);
  }
  
  return(r);
}

// ################################################################################
// ############################# MCMC process #####################################
// ################################################################################

// Ok, here's the big code that runs the MCMC process:
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;
#include <limits>

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List mcmc_binom_C(vec counts, vec total_obs, List start, int iterations, mat covar1,
                  List step, List prior, int burnin, int nth) {              
  
  // start is the same size as a single iteration of the process:
  List out(2 * start.size());
  
  // Creating the distance matrix, unchanged throughout
  List dist = dist_mat(covar1.cols(1, 2));
  
  //// Forming the matrices for zeta, to be filled during the MCMC step:
  // The zeta parameter:
  vec start0 = start[0];
  mat zeta;
  zeta.insert_rows(0, start0.t());
  
  // The alpha_zeta coefficients:
  vec start1 = start[1];
  mat alpha_zeta;
  alpha_zeta.insert_rows(0, start1.t());
  
  // The beta_zeta matrix:
  vec start2 = start[2];
  mat beta_zeta;
  beta_zeta.insert_rows(0, start2.t());
  
  // The sigsq_zeta coefficient:
  vec start5 = start[3];
  double s5 = as_scalar(start5);
  NumericVector sigsq_zeta;
  sigsq_zeta.push_back(s5);
  
  // The tausq_zeta coefficient:
  vec start6 = start[4];
  double s6 = as_scalar(start6);
  NumericVector tausq_zeta;
  tausq_zeta.push_back(s6);
  
  // Setting the starting values for zeta and the zeta hyper-parameters:
  vec prop_zeta = zeta.row(0).t();
  vec prop_alpha_zeta = alpha_zeta.row(0).t();
  mat prop_beta_zeta(2, 2);
  prop_beta_zeta.row(0) = beta_zeta.submat(0, 0, 0, 1); // A complicated way of getting at the elements needed
  prop_beta_zeta.row(1) = beta_zeta.submat(0, 2, 0, 3);
  double prop_sigsq_zeta = as_scalar(sigsq_zeta(0));
  double prop_tausq_zeta = as_scalar(tausq_zeta(0));
  
  // Now creating vectors to keep track of the acceptance rates
  int n1 = prop_zeta.size(); int n2 = prop_alpha_zeta.size();
  int n3 = prop_beta_zeta.size();
  Rcpp::NumericVector tempr(n1); Rcpp::NumericVector temps(n2);
  Rcpp::NumericVector tempt(n3);
  
  // These are to be filled in:
  Rcpp::NumericVector acc_zeta(n1);
  Rcpp::NumericVector acc_alpha_zeta(n2);
  vec acc_beta_zeta(n3);
  double acc_sigsq_zeta = 0;
  double acc_tausq_zeta = 0;
  bool b1 = 0; bool b2 = 0;
  Rcpp::NumericMatrix temp_acc_beta_zeta(n3/2, n3/2);
  
  /////////////////////////////////////////////////////////////////////////////////////////
  // Calculating Sigma_zeta for input to the first step of the MCMC process:
  // Also creating a copy, to have two floating around at each stage
  // (Saving two big matrices means it won't need to be recreated each time - a worthwhile saving)
  // Also calculating the inverse, and a copy of it for the same reason
  mat Sigma_zeta = get_mat_sigma(dist, prop_beta_zeta, prop_sigsq_zeta, prop_tausq_zeta);
  mat Sigma_new_zeta = Sigma_zeta;
  mat InvSigma_zeta = inv_sympd(Sigma_zeta);
  mat InvSigma_new_zeta = InvSigma_zeta;
  
  ////////////////////////////////////////////////////////////////////////////////////////
  // The identity for use in updating tausq:
  mat Inxn;
  Inxn.copy_size(Sigma_zeta);
  Inxn.eye();
  
  // doubles needed for the log det. step, which will be calculated once in each loop:
  // Again, this saves a calculation for each gridpoint *within* each iteration, a large saving
  double dett_zeta; double sign;
  log_det(dett_zeta, sign, Sigma_zeta);
  double dettnew_zeta = 0;
  
  // This command below will stop the code immediately if
  // start values inconsistent with the data are used
  double a = std::numeric_limits<double>::infinity();
  if ( dbinomC(counts, total_obs, zeta.row(0).t()) == -a ||
       gp_mvnC(zeta.row(0).t(), InvSigma_zeta, alpha_zeta.row(0).t(), covar1, dett_zeta) == -a )
  {
    cout << "Stop. The starting values are inconsistent with the data." << endl;
  }
  
  // Picking out the steps for the zeta updates:
  // (These are essentially the standard deviations, inputted at the beginning. Adjusting these
  // will affect the acceptance rate. Too big, and not enough values are accepted.
  // Too small, and too many values are accepted, leading to slow mixing (exploration) of
  // the posterior space)
  vec step_zeta = step[0]; vec step_alpha_zeta = step[1]; vec step_sigsq_zeta1 = step[2];
  double step_sigsq_zeta = as_scalar(step_sigsq_zeta1); vec step_tausq_zeta1 = step[3];
  double step_tausq_zeta = as_scalar(step_tausq_zeta1);
  
  // Picking out the zeta hyper parameters:
  // These are the priors on the layer 3 parameters, and the end of our hierarchy:
  vec alpha_zeta_hyper_mean = prior[0]; mat alpha_zeta_hyper_sd = prior[1]; vec beta_zeta_hyper_prior = prior[2];
  // beta is a discrete parameter, and is handled differently:
  int lenb_zeta = beta_zeta_hyper_prior.size();
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // The zeta hyper parameters, uncomment when is is assumed constant:
  // vec temp_mean = prior[6]; double zeta_hyper_mean = as_scalar(temp_mean);
  // vec temp_sd = prior[7]; double zeta_hyper_sd = as_scalar(temp_sd);
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  // Defining the items to be used in the zeta update:
  // These will essentially be the last items in the chain - vectors and scalars that are rewritten at
  // each line, and copied into the big matrix if they are accepted:
  vec newprop_zeta;
  vec newprop_alpha_zeta;
  mat newprop_beta_zeta; int nb; int c;
  double temp_sigsq; double newprop_sigsq_zeta;
  double temp_tausq; double newprop_tausq_zeta;
  
  // Doubles, vecs and ints to be used within the loops:
  double rndm;
  double probab;
  vec rnd;
  int m;
  
  ////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// MCMC piece ///////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  // Now for the MCMC step, now that everything has been set up:
  for(int i = 0; i < iterations; ++i) {
    
    // At the i^th step, all will be accepted or rejected with this probability:
    rnd = runif(1, 0, 1);
    rndm = as_scalar(rnd);
    
    // Calculating the log(det), dett, which will be used in the calls to the mvn:
    log_det(dett_zeta, sign, Sigma_zeta);
    
    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////// Zeta updates now: //////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    // Updating the zeta parameter:
    newprop_zeta = prop_zeta;
    for (int l = 0; l < prop_zeta.size(); ++l) {
      newprop_zeta[l] = rnormscalarC(prop_zeta[l], step_zeta(0));
      probab = exp( post_zetaC(counts, total_obs, newprop_zeta, prop_alpha_zeta, 
                               covar1, InvSigma_zeta, l, dett_zeta) -
                                 post_zetaC(counts, total_obs, prop_zeta, prop_alpha_zeta,
                                            covar1, InvSigma_zeta, l, dett_zeta) );
      
      // Accept or reject step:
      if (rndm < probab) {
        prop_zeta = newprop_zeta;
        tempr[l] = 1;
      } else {
        newprop_zeta = prop_zeta;
        tempr[l] = 0;
      }
    }
    
    // Updating the alpha_zeta parameter:
    newprop_alpha_zeta = prop_alpha_zeta;
    for (int l = 0; l < prop_alpha_zeta.size(); ++l) {
      
      newprop_alpha_zeta[l] = rnormscalarC(prop_alpha_zeta[l], step_alpha_zeta[l]);
      
      probab = exp (post_alpha_zetaC(newprop_alpha_zeta, prop_zeta, covar1, InvSigma_zeta,
                                     alpha_zeta_hyper_mean, alpha_zeta_hyper_sd, dett_zeta) -
                                       post_alpha_zetaC(prop_alpha_zeta, prop_zeta, covar1, InvSigma_zeta,
                                                        alpha_zeta_hyper_mean, alpha_zeta_hyper_sd, dett_zeta) ) ;
      
      //  Accept or reject step:
      if (rndm < probab) {
        prop_alpha_zeta = newprop_alpha_zeta;
        temps[l] = 1;
      } else {
        newprop_alpha_zeta = prop_alpha_zeta;
        temps[l] = 0;
      }
    }
    
    // Updating the beta_zeta parameter:  
    nb = prop_beta_zeta.n_rows;
    // Reshaping the vector into a matrix:
    newprop_beta_zeta = reshape(prop_beta_zeta, nb, nb);
    
    // A double loop running over the lower diagonal part of the matrix:
    for(int k=0; k<nb; ++k) {
      for(int p=0; p<=k; ++p) {
        c = rand() % lenb_zeta;
        newprop_beta_zeta(k, p) = beta_zeta_hyper_prior(c); // select the new value
        newprop_beta_zeta(p, k) = newprop_beta_zeta(k, p); // mirror across the diagonal
        
        // Checking that the proposed matrix is positive definite and invertible before updating it by MCMC:
        if (pos_def_testC(newprop_beta_zeta) == 1) {
          Sigma_new_zeta = get_mat_sigma(dist, newprop_beta_zeta, prop_sigsq_zeta, prop_tausq_zeta);
          InvSigma_new_zeta = inv_sympd(Sigma_new_zeta);
          log_det(dettnew_zeta, sign, Sigma_new_zeta);
          
          // And if it is, calculating the ratio with the new Sigma and beta vs. the old ones:
          probab = exp (post_beta_zetaC(InvSigma_new_zeta, newprop_beta_zeta, prop_zeta,
                                        prop_alpha_zeta, covar1, dettnew_zeta) -
                                          post_beta_zetaC(InvSigma_zeta, prop_beta_zeta, prop_zeta,
                                                          prop_alpha_zeta, covar1, dett_zeta) );
          
          // Accept or reject step:
          if (rndm < probab) {
            prop_beta_zeta = newprop_beta_zeta;
            Sigma_zeta = Sigma_new_zeta;
            InvSigma_zeta = InvSigma_new_zeta;
            dett_zeta = dettnew_zeta;
            temp_acc_beta_zeta(k, p) = 1;
            temp_acc_beta_zeta(p, k) = 1;
          } else {
            newprop_beta_zeta = prop_beta_zeta;
            Sigma_new_zeta = Sigma_zeta;
            InvSigma_new_zeta = InvSigma_zeta;
            dettnew_zeta = dett_zeta;
            temp_acc_beta_zeta(k, p) = 0;
            temp_acc_beta_zeta(p, k) = 0;
          }
        }
      }
    }                                            
    
    // Updating the sigsq_zeta parameter now:
    temp_sigsq = rnormscalarC(log(prop_sigsq_zeta), step_sigsq_zeta);
    newprop_sigsq_zeta = exp(temp_sigsq);
    
    // Updating Sigma:
    Sigma_new_zeta = get_mat_sigma(dist, prop_beta_zeta, newprop_sigsq_zeta, prop_tausq_zeta);
    
    InvSigma_new_zeta = inv_sympd(Sigma_new_zeta);
    log_det(dettnew_zeta, sign, Sigma_new_zeta);
    
    // Calculating probab:
    probab = exp( post_sigsq_zetaC(InvSigma_new_zeta, prop_zeta, prop_alpha_zeta,
                                   covar1, newprop_sigsq_zeta, dettnew_zeta) -
                                     post_sigsq_zetaC(InvSigma_zeta, prop_zeta, prop_alpha_zeta,
                                                      covar1, prop_sigsq_zeta, dett_zeta) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_sigsq_zeta = newprop_sigsq_zeta;
      Sigma_zeta = Sigma_new_zeta;
      InvSigma_zeta = InvSigma_new_zeta;
      dett_zeta = dettnew_zeta;
      b1 = 1;
    } else {
      newprop_sigsq_zeta = prop_sigsq_zeta;
      Sigma_new_zeta = Sigma_zeta;
      InvSigma_new_zeta = InvSigma_zeta;
      dettnew_zeta = dett_zeta;
      b1 = 0;
    }    
    
    // Updating the tausq_zeta parameter now:
    temp_tausq = rnormscalarC(log(prop_tausq_zeta), step_tausq_zeta);
    newprop_tausq_zeta = exp(temp_tausq);
    
    // Updating Sigma:
    Sigma_new_zeta = get_mat_sigma(dist, prop_beta_zeta, prop_sigsq_zeta, newprop_tausq_zeta);
    
    InvSigma_new_zeta = inv_sympd(Sigma_new_zeta);
    log_det(dettnew_zeta, sign, Sigma_new_zeta);
    
    probab = exp( post_tausq_zetaC(InvSigma_new_zeta, prop_zeta, prop_alpha_zeta,
                                   covar1, newprop_tausq_zeta, dettnew_zeta) -
                                     post_tausq_zetaC(InvSigma_zeta, prop_zeta, prop_alpha_zeta,
                                                      covar1, prop_tausq_zeta, dett_zeta) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_tausq_zeta = newprop_tausq_zeta;
      Sigma_zeta = Sigma_new_zeta;
      InvSigma_zeta = InvSigma_new_zeta;
      dett_zeta = dettnew_zeta;
      b2 = 1;
    } else {
      newprop_tausq_zeta = prop_tausq_zeta;
      Sigma_new_zeta = Sigma_zeta;
      InvSigma_new_zeta = InvSigma_zeta;
      dettnew_zeta = dett_zeta;
      b2 = 0;
    }    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////// As an alternative, where zeta is considered as a constant /////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Updating the zeta parameter as a constant now:
    //    newprop_zeta = prop_zeta;
    //    for (int l = 0; l < prop_zeta.size(); ++l) {
    //      newprop_zeta[l] = rnormscalarC(prop_zeta[l], step_zeta(0));
    //      probab = exp( post_zetaC2(counts, total_obs, newprop_zeta, l) - //+
    //                            //dnormscalarC(newprop_zeta[l], zeta_hyper_mean, zeta_hyper_sd) -
    //                          post_shapeC2(counts, total_obs, prop_zeta, l) ); //+
    //                            //dnormscalarC(as_scalar(prop_zeta[l]), zeta_hyper_mean, zeta_hyper_sd) );
    //
    //    //  Come back to this:    if(is.na(probab)) {probab <- 0}
    //      if (rndm < probab) {
    //        prop_zeta = newprop_zeta;
    //      } else {
    //        newprop_zeta = prop_zeta;
    //      }
    //    }
    //    
    ///////////////////////////////////////////////////////////////////////////////////////////////    
    //////////////////////////////// Comment out one or other pieces above ////////////////////////
    
    //////////////////////////////// Now to calculate the acceptance rates: ///////////////////////
    acc_zeta = acc_zeta + tempr;
    acc_alpha_zeta = acc_alpha_zeta + temps;
    vec temp = as<vec>(temp_acc_beta_zeta);
    acc_beta_zeta = acc_beta_zeta + temp;
    acc_sigsq_zeta = acc_sigsq_zeta + b1;
    acc_tausq_zeta = acc_tausq_zeta + b2;
    
    /////////////////// Saving these at selected iterations: //////////////////////////////////////
    // Save if we're beyond the burn-in period *and* it's every nth iteration:
    if(i > burnin && i % nth == 0) {
      
      // Printing the value of i
      Rprintf("%d \n", i);

      m = zeta.n_rows;
      
      // Writing all the values at each selected cycle for zeta:
      zeta.insert_rows(m, prop_zeta.t());
      alpha_zeta.insert_rows(m, prop_alpha_zeta.t());
      beta_zeta.insert_rows(m, vectorise(prop_beta_zeta).t());
      sigsq_zeta.push_back(prop_sigsq_zeta);
      tausq_zeta.push_back(prop_tausq_zeta);
      
    }
  }

  //////////////////////////// Getting ready to output results: //////////////////////
  // Writing each element of the list:
  out[0] = zeta;
  out[1] = alpha_zeta;
  out[2] = beta_zeta;
  out[3] = sigsq_zeta;
  out[4] = tausq_zeta;
  
  out[5] = acc_zeta/iterations;
  out[6] = acc_alpha_zeta/iterations;
  out[7] = acc_beta_zeta/iterations; // Because updates are *discrete*, sometimes a 'new' value, the same as the old
  // is accepted, leading to slightly inflated acceptance rates here
  out[8] = acc_sigsq_zeta/iterations;
  out[9] = acc_tausq_zeta/iterations;
  
  // Creating the names for all the elements of the output list:
  int g1 = out.size();
  CharacterVector names(g1);
  names[0] = "zeta";
  names[1] = "alpha_zeta";
  names[2] = "beta_zeta";
  names[3] = "sigsq_zeta";
  names[4] = "tausq_zeta";
  
  names[5] = "acc_zeta";
  names[6] = "acc_alpha_zeta";
  names[7] = "acc_beta_zeta";
  names[8] = "acc_sigsq_zeta";
  names[9] = "acc_tausq_zeta";
  
  out.attr("names") = names;
  
  return out;
}

