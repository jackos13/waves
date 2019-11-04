//////// Cpp GPD Script - Version 3 ////////////
//////// This script is a tidied up version of the first two versions
//////// It includes a free shape parameter (not constrained
//////// to be positive) and a count of the total acceptance rate
// ##################### Functions needed for model 1 - the GPD ###################
// ################################################################################

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calculating the distance list:
// This function takes an n x 2 matrix and returns a list. Each component is an n x 2 matrix showing the distance
// between its row and all the other rows. e.g. element 1 in the list has its first row as the distance between
// point 1 and point 1, the second row as the distance between point 1 and point 2 etc.
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
      sig(i, j) = sigsq * exp( - as_scalar( tmpi.row(j) * invb * tmpi.row(j).t() ) );
      if (i == j) {
        sig(i, j) = sig(i, j) + tausq;
      } else {
        sig(j, i) = sig(i, j);
      }   
    }
  }
  
  return sig;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ################################################################################
// ############################# Data layer #######################################
// ################################################################################
// The full GPD, works on a list:
// This function, on the log scale, takes in a list of excesses for each gridpoint, and a single value
// of the threshold, the scale and the shape for each gridpoint,
// and then evaluates the gpd at each. These are then summed up (log(product) = sum(logs) etc.)
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double dgpdC(List data, vec mu, vec lscale, vec shape) {
  int n = data.size();
  List out(n);
  vec sums(n);

  // Exponentiating the log of the scale:
  vec scale = exp(lscale);
  
  for(int i = 0; i < n; ++i) {
    vec tmp = data[i];
    int m = data.size();
    out[i] = tmp - mu[i];
  
  // This condition isn't needed now, but I'll leave it in in case I 
  // need to change it again
    if(shape[i]==0) {
      out[i] = log (1/scale[i]) - (tmp - mu[i])/scale[i]; 
    } else {
      out[i] = log(1/scale[i]) + log( pow (1 + shape[i] * (tmp - mu[i])/scale[i], (-1/shape[i]) - 1));
    }
    sums[i] = sum(as<vec>(out[i]));
  }
  
  return sum(sums);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The GPD, works on a single point in space:
// This followed the realisation that when I update a single value of the scale say, I only need to evaluate the gpd
// component of that at the particular gridpoint
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double dgpd2C(vec data, double mu, double lscale, double shape) {
  
  int n = data.size();
  vec out(n);

  // Exponentiating the log of the scale:
  double scale = exp(lscale);
  
  double sums;
  
  // There are two cases: shape=0 and shape=/=0:
  if(shape==0) {
    out = -log(scale) - (data - mu)/scale; 
  } else {
    out = -log(scale) + (-1/shape - 1) * log( 1 + shape * (data - mu)/scale );
  }
  sums = sum(out);
  
  return sums;
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
// Here, the precision is used - where it's calculated as the inverse of sigma directly withing
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
double gp_mvnC(vec x, mat invsigma, vec alpha, mat covar, double dett) {
  
  vec mean = covar * alpha;
  
  double out = mvnC2(x, mean, invsigma, dett);
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
// Evaluating the posterior conditional of the scale:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_scaleC(List data, vec mu, vec lscale, vec shape, vec alpha, mat covar,
                                                  mat invsigma, int l, double dett_scale) {
  double a = dgpd2C(data[l], mu(l), lscale(l), shape(l));
  double b = gp_mvnC(lscale, invsigma, alpha, covar, dett_scale);
  return a + b;
}

// ################################################################################
// Evaluating the posterior conditional of the shape:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_shapeC(List data, vec mu, vec lscale, vec shape, vec alpha, mat covar,
                                mat invsigma_shape, int l, double dett_shape) {
  double a = dgpd2C(data[l], mu[l], lscale[l], shape[l]);
  double b = gp_mvnC(shape, invsigma_shape, alpha, covar, dett_shape);
  return a + b;
}

// ################################################################################
// Evaluating the posterior conditional of the alpha_scale parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_alpha_scaleC(vec alpha_scale, vec lscale, mat covar, mat invsigma,
                              vec alpha_scale_hyper_mean, mat alpha_scale_hyper_sd, double dett_scale) {
  double a = gp_mvnC(lscale, invsigma, alpha_scale, covar, dett_scale); // Layer2
  double b = mvnC(alpha_scale, alpha_scale_hyper_mean, alpha_scale_hyper_sd) ; //Layer3
  return a + b;
}  

// ################################################################################
// Evaluating the posterior conditional of the beta_scale parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_beta_scaleC(mat invsigma, mat beta_scale, vec lscale, vec alpha_scale, mat covar, double dett_scale) {
  double a = gp_mvnC(lscale, invsigma, alpha_scale, covar, dett_scale); // Layer2
  //double b =  ; //Layer3 - blank for now
  return a;
}  

// ################################################################################
// Evaluating the posterior conditional of the sigsq_scale parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_sigsq_scaleC(mat invsigma, vec lscale, vec alpha, mat covar, double sigsq, double dett_scale,
                          double sigsq_scale_mean, double sigsq_scale_sd) {
  double a = gp_mvnC(lscale, invsigma, alpha, covar, dett_scale); // Layer2
  double b = dnormscalarC(log(sigsq), sigsq_scale_mean, sigsq_scale_sd); // Layer3
  //double b =  ; //Layer3 - blank for now
  return a + b;
}

// ################################################################################
// Evaluating the posterior conditional of the tausq_scale parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_tausq_scaleC(mat invsigma, vec lscale, vec alpha, mat covar, double tausq, double dett_scale,
                          double tausq_scale_mean, double tausq_scale_sd) {
  double a = gp_mvnC(lscale, invsigma, alpha, covar, dett_scale); // Layer2
  double b = dnormscalarC(log(tausq), tausq_scale_mean, tausq_scale_sd);
  //double b =  ; //Layer3 - blank for now
  return a + b;
}

// ################################################################################
// Evaluating the posterior conditional of the alpha_shape parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_alpha_shapeC(vec alpha_shape, vec shape, mat covar, mat invsigma_shape,
                          vec alpha_shape_hyper_mean, mat alpha_shape_hyper_sd, double dett_shape) {
  double a = gp_mvnC(shape, invsigma_shape, alpha_shape, covar, dett_shape); // Layer2
  double b = mvnC(alpha_shape, alpha_shape_hyper_mean, alpha_shape_hyper_sd) ; //Layer3
  return a + b;
}  

// ################################################################################
// Evaluating the posterior conditional of the beta_shape parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_beta_shapeC(mat invsigma_shape, mat beta_shape, vec shape, vec alpha_shape,
                              mat covar, double dett_shape) {
  double a = gp_mvnC(shape, invsigma_shape, alpha_shape, covar, dett_shape); // Layer2
  //double b =  ; //Layer3 - blank for now
  return a;
}  

// ################################################################################
// Evaluating the posterior conditional of the sigsq_shape parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_sigsq_shapeC(mat invsigma_shape, vec shape, vec alpha, mat covar,
                                  double sigsq, double dett_shape,
                                  double sigsq_shape_mean, double sigsq_shape_sd) {
  double a = gp_mvnC(shape, invsigma_shape, alpha, covar, dett_shape); // Layer2
  double b = dnormscalarC(log(sigsq), sigsq_shape_mean, sigsq_shape_sd); // Layer3
  //double b =  ; //Layer3 - blank for now
  return a + b;
}

// ################################################################################
// Evaluating the posterior conditional of the tausq_shape parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_tausq_shapeC(mat invsigma_shape, vec shape, vec alpha, mat covar,
                                      double tausq, double dett_shape,
                                      double tausq_shape_mean, double tausq_shape_sd) {
  double a = gp_mvnC(shape, invsigma_shape, alpha, covar, dett_shape); // Layer2
  double b = dnormscalarC(log(tausq), tausq_shape_mean, tausq_shape_sd); // Layer3
  //double b =  ; //Layer3 - blank for now
  return a + b;
}

// ################################################################################

// Alternatives to the GP here, for comparison: Assuming the scale is a constant value:

// Evaluating the full posterior conditional of the scale, where it is assumed constant:
// (Can include a prior on this too, by uncommenting below)
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_scaleC2(List data, vec mu, vec lscale, vec shape, int l) {
  double a = dgpd2C(data[l], mu[l], lscale[l], shape[l]);
//  double b = dnormscalarC(shape, prior_mean, prior_sd);
  return a;
}

// ################################################################################
// Evaluating the posterior conditional of the shape, where it is assumed constant:
// (Can include a prior on this too, by uncommenting below)
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_shapeC2(List data, vec mu, vec lscale, vec shape, int l) {
  double a = dgpd2C(data[l], mu[l], lscale[l], shape[l]);
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
List mcmcC(List data, vec mu, List start, int iterations, mat covar1, mat covar2,
                                          List step, List prior, int burnin, int nth) {              
  
  // start is the same size as a single iteration of the process:
  List out(2 * start.size() + 2);
  
  // Creating the distance matrix, unchanged throughout
  List dist = dist_mat(covar1.cols(1, 2));
  
  //// Forming the matrices for the scale, to be filled during the MCMC step:
  // The log(scale) parameter:
  vec start0 = start[0];
  mat lscale;
  lscale.insert_rows(0, start0.t());
  
  // The alpha_scale coefficients:
  vec start1 = start[1];
  mat alpha_scale;
  alpha_scale.insert_rows(0, start1.t());

  // The beta_scale matrix:
  vec start2 = start[2];
  mat beta_scale;
  beta_scale.insert_rows(0, start2.t());

  // The sigsq_scale coefficient:
  vec start5 = start[3];
  double s5 = as_scalar(start5);
  NumericVector sigsq_scale;
  sigsq_scale.push_back(s5);

  // The tausq_scale coefficient:
  vec start6 = start[4];
  double s6 = as_scalar(start6);
  NumericVector tausq_scale;
  tausq_scale.push_back(s6);

  //// Next, forming the matrices for the shape, to be filled during the MCMC step:
  // The log(shape) parameter:
  vec start7 = start[5];
  mat shape;
  shape.insert_rows(0, start7.t());

  // The alpha_shape coefficients:
  vec start8 = start[6];
  mat alpha_shape;
  alpha_shape.insert_rows(0, start8.t());

  // The beta_shape matrix:
  vec start9 = start[7];
  mat beta_shape;
  beta_shape.insert_rows(0, start9.t());

  // The sigsq_shape coefficient:
  vec start12 = start[8];
  double s12 = as_scalar(start12);
  NumericVector sigsq_shape;
  sigsq_shape.push_back(s12);

  // The tausq_shape coefficient:
  vec start13 = start[9];
  double s13 = as_scalar(start13);
  NumericVector tausq_shape;
  tausq_shape.push_back(s13);

  // Setting the starting values for the scale and scale hyper-parameters:
  vec prop_lscale = lscale.row(0).t();
  vec prop_alpha_scale = alpha_scale.row(0).t();
  mat prop_beta_scale(2, 2);
  prop_beta_scale.row(0) = beta_scale.submat(0, 0, 0, 1); // A complicated way of getting at the elements needed
  prop_beta_scale.row(1) = beta_scale.submat(0, 2, 0, 3);
  double prop_sigsq_scale = as_scalar(sigsq_scale(0));
  double prop_tausq_scale = as_scalar(tausq_scale(0));

  // Setting the starting values for the shape:
  vec prop_shape = shape.row(0).t();
  vec prop_alpha_shape = alpha_shape.row(0).t();
  mat prop_beta_shape(2, 2);
  prop_beta_shape.row(0) = beta_shape.submat(0, 0, 0, 1);
  prop_beta_shape.row(1) = beta_shape.submat(0, 2, 0, 3);
  double prop_sigsq_shape = as_scalar(sigsq_shape(0));
  double prop_tausq_shape = as_scalar(tausq_shape(0));

  // Now creating vectors to keep track of the acceptance rates
  int n1 = prop_lscale.size(); int n2 = prop_alpha_scale.size();
  int n3 = prop_beta_scale.size();
  Rcpp::NumericVector tempr(n1); Rcpp::NumericVector temps(n2);
  Rcpp::NumericVector tempt(n3);

  int m1 = prop_shape.size(); int m2 = prop_alpha_shape.size();
  int m3 = prop_beta_shape.size();
  Rcpp::NumericVector tempu(m1); Rcpp::NumericVector tempv(m2);
  Rcpp::NumericVector tempw(m3);

  // For finding the acceptance rates:
  // These are to be filled in:
  Rcpp::NumericVector acc_lscale(n1);
  Rcpp::NumericVector acc_alpha_scale(n2);
  vec acc_beta_scale(n3);
  double acc_sigsq_scale = 0;
  double acc_tausq_scale = 0;
  bool b1 = 0; bool b2 = 0;
  Rcpp::NumericMatrix temp_acc_beta_scale(n3/2, n3/2);

  Rcpp::NumericVector acc_shape(m1);
  Rcpp::NumericVector acc_alpha_shape(m2);
  vec acc_beta_shape(m3);
  double acc_sigsq_shape = 0;
  double acc_tausq_shape = 0;
  bool b3 = 0; bool b4 = 0;
  Rcpp::NumericMatrix temp_acc_beta_shape(m3/2, m3/2);

  /////////////////////////////////////////////////////////////////////////////////////////
  // Calculating Sigma_scale for input to the first step of the MCMC process:
  // Also creating a copy, to have two floating around at each stage
  // (Saving two big matrices means it won't need to be recreated each time - a worthwhile saving)
  // Also calculating the inverse, and a copy of it for the same reason
  mat Sigma_scale = get_mat_sigma(dist, prop_beta_scale, prop_sigsq_scale, prop_tausq_scale);
  mat Sigma_new_scale = Sigma_scale;
  mat InvSigma_scale = inv(Sigma_scale);
  mat InvSigma_new_scale = InvSigma_scale;

  /////////////////////////////////////////////////////////////////////////////////////////
  // Calculating Sigma_shape for input to the first step of the MCMC process:
  // And a copy, and an inverse, and a copy, for the reasons above:
  mat Sigma_shape = get_mat_sigma(dist, prop_beta_shape, prop_sigsq_shape, prop_tausq_shape);
  mat Sigma_new_shape = Sigma_shape;
  mat InvSigma_shape = inv(Sigma_shape);
  mat InvSigma_new_shape = InvSigma_shape;

  ////////////////////////////////////////////////////////////////////////////////////////
  // The identity for use in updating tausq:
  mat Inxn;
  Inxn.copy_size(Sigma_scale);
  Inxn.eye();

  // doubles needed for the log det. step, which will be calculated once in each loop:
  // Again, this saves a calculation for each gridpoint *within* each iteration, a large saving
  double dett_scale; double sign;
  log_det(dett_scale, sign, Sigma_scale);
  double dettnew_scale = 0;

  double dett_shape;
  log_det(dett_shape, sign, Sigma_shape);
  double dettnew_shape = 0;

  // This command below will stop the code immediately if
  // start values inconsistent with the data are used
  double a = std::numeric_limits<double>::infinity();
  if ( dgpdC(data, mu, lscale.row(0).t(), shape.row(0).t()) == -a ||
        gp_mvnC(lscale.row(0).t(), InvSigma_scale, alpha_scale.row(0).t(), covar1, dett_scale) == -a ||
        gp_mvnC(shape.row(0).t(), InvSigma_shape, alpha_shape.row(0).t(), covar2, dett_shape) == -a )
        {
          cout << "Stop. The starting values are inconsistent with the data." << endl;
        }

  // Picking out the steps for the scale updates:
  // (These are essentially the standard deviations, inputted at the beginning. Adjusting these
  // will affect the acceptance rate. Too big, and not enough values are accepted.
  // Too small, and too many values are accepted, leading to slow mixing (exploration) of
  // the posterior space)
  vec step_scale = step[0]; vec step_alpha_scale = step[1]; vec step_sigsq_scale1 = step[2];
  double step_sigsq_scale = as_scalar(step_sigsq_scale1); vec step_tausq_scale1 = step[3];
  double step_tausq_scale = as_scalar(step_tausq_scale1);

  // Picking out the steps for the shape updates:
  // As above. The single values need to be taken as a vector, then set to a scalar:
  vec step_shape = step[4]; vec step_alpha_shape = step[5]; vec step_sigsq_shape1 = step[6];
  double step_sigsq_shape = as_scalar(step_sigsq_shape1); vec step_tausq_shape1 = step[7];
  double step_tausq_shape = as_scalar(step_tausq_shape1);

  // Picking out the scale hyper parameters:
  // These are the priors on the layer 3 parameters, and the end of our hierarchy:
  vec alpha_scale_hyper_mean = prior[0]; mat alpha_scale_hyper_sd = prior[1]; vec beta_scale_hyper_prior = prior[2];
  // beta is a discrete parameter, and is handled differently:
  int lenb_scale = beta_scale_hyper_prior.size();
  double sigsq_scale_mean = prior[8]; double sigsq_scale_sd = prior[9];
  double tausq_scale_mean = prior[10]; double tausq_scale_sd = prior[11];

  // Picking out the shape hyper parameters:
  vec alpha_shape_hyper_mean = prior[3]; mat alpha_shape_hyper_sd = prior[4]; vec beta_shape_hyper_prior = prior[5];
  // beta is a discrete parameter, and is handled differently:
  int lenb_shape = beta_shape_hyper_prior.size();
  double sigsq_shape_mean = prior[12]; double sigsq_shape_sd = prior[13];
  double tausq_shape_mean = prior[14]; double tausq_shape_sd = prior[15];

  ///////////////////////////////////////////////////////////////////////////////////////////
  // The shape hyper parameters, when is is assumed constant:
  // (This can be added to for a constant scale too)
  vec temp_mean = prior[6]; double shape_hyper_mean = as_scalar(temp_mean);
  vec temp_sd = prior[7]; double shape_hyper_sd = as_scalar(temp_sd);
  ///////////////////////////////////////////////////////////////////////////////////////////

  // Defining the items to be used in the scale update:
  // These will essentially be the last items in the chain - vectors and scalars that are rewritten at
  // each line, and copied into the big matrix if they are accepted:
  vec newprop_lscale;
  vec newprop_alpha_scale;
  mat newprop_beta_scale; int nb; int c;
  double temp_sigsq; double newprop_sigsq_scale;
  double temp_tausq; double newprop_tausq_scale;

  // Defining the items to be used in the shape update:
  vec newprop_shape;
  vec newprop_alpha_shape;
  mat newprop_beta_shape;
  double newprop_sigsq_shape;
  double newprop_tausq_shape;

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
    log_det(dett_scale, sign, Sigma_scale);
    log_det(dett_shape, sign, Sigma_shape);

    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////// Scale updates now: //////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    // Updating the lscale parameter:
    newprop_lscale = prop_lscale;
    for (int l = 0; l < prop_lscale.size(); ++l) {
      newprop_lscale[l] = rnormscalarC(prop_lscale[l], step_scale(0));
      probab = exp( post_scaleC(data, mu, newprop_lscale, prop_shape, prop_alpha_scale,
                                  covar1, InvSigma_scale, l, dett_scale) -
                      post_scaleC(data, mu, prop_lscale, prop_shape, prop_alpha_scale,
                                  covar1, InvSigma_scale, l, dett_scale) );

    //  Accept or reject step:
      if (rndm < probab) {
        prop_lscale = newprop_lscale;
        tempr[l] = 1;
      } else {
        newprop_lscale = prop_lscale;
        tempr[l] = 0;
      }
    }

   // Updating the alpha_scale parameter:
    newprop_alpha_scale = prop_alpha_scale;
    for (int l = 0; l < prop_alpha_scale.size(); ++l) {

      newprop_alpha_scale[l] = rnormscalarC(prop_alpha_scale[l], step_alpha_scale[l]);

      probab = exp (post_alpha_scaleC(newprop_alpha_scale, prop_lscale, covar1, InvSigma_scale,
                                            alpha_scale_hyper_mean, alpha_scale_hyper_sd, dett_scale) -
                          post_alpha_scaleC(prop_alpha_scale, prop_lscale, covar1, InvSigma_scale,
                                            alpha_scale_hyper_mean, alpha_scale_hyper_sd, dett_scale) ) ;

    //  Accept or reject step:
      if (rndm < probab) {
        prop_alpha_scale = newprop_alpha_scale;
        temps[l] = 1;
      } else {
        newprop_alpha_scale = prop_alpha_scale;
        temps[l] = 0;
      }
    }

    // Updating the beta_scale parameter:
    nb = prop_beta_scale.n_rows;
    // Reshaping the vector into a matrix:
    newprop_beta_scale = reshape(prop_beta_scale, nb, nb);

    // A double loop running over the lower diagonal part of the matrix:
    for(int k=0; k<nb; ++k) {
      for(int p=0; p<=k; ++p) {
        c = rand() % lenb_scale;
        newprop_beta_scale(k, p) = beta_scale_hyper_prior(c); // select the new value
        newprop_beta_scale(p, k) = newprop_beta_scale(k, p); // mirror across the diagonal

        // Checking that the proposed matrix is positive definite and invertible before updating it by MCMC:
        if (pos_def_testC(newprop_beta_scale) == 1) {
          Sigma_new_scale = get_mat_sigma(dist, newprop_beta_scale, prop_sigsq_scale, prop_tausq_scale);
          InvSigma_new_scale = inv(Sigma_new_scale);
          log_det(dettnew_scale, sign, Sigma_new_scale);

          // And if it is, calculating the ratio with the new Sigma and beta vs. the old ones:
          probab = exp (post_beta_scaleC(InvSigma_new_scale, newprop_beta_scale, prop_lscale,
                                                                prop_alpha_scale, covar1, dettnew_scale) -
                          post_beta_scaleC(InvSigma_scale, prop_beta_scale, prop_lscale,
                                                                prop_alpha_scale, covar1, dett_scale) );

          // Accept or reject step:
          if (rndm < probab) {
            prop_beta_scale = newprop_beta_scale;
            Sigma_scale = Sigma_new_scale;
            InvSigma_scale = InvSigma_new_scale;
            dett_scale = dettnew_scale;
            temp_acc_beta_scale(k, p) = 1;
            temp_acc_beta_scale(p, k) = 1;
          } else {
            newprop_beta_scale = prop_beta_scale;
            Sigma_new_scale = Sigma_scale;
            InvSigma_new_scale = InvSigma_scale;
            dettnew_scale = dett_scale;
            temp_acc_beta_scale(k, p) = 0;
            temp_acc_beta_scale(p, k) = 0;
          }
        }
      }
    }

    // Updating the sigsq_scale parameter now:
    temp_sigsq = rnormscalarC(log(prop_sigsq_scale), step_sigsq_scale);
    newprop_sigsq_scale = exp(temp_sigsq);

    // Updating Sigma:
    Sigma_new_scale = get_mat_sigma(dist, prop_beta_scale, newprop_sigsq_scale, prop_tausq_scale);

    InvSigma_new_scale = inv(Sigma_new_scale);
    log_det(dettnew_scale, sign, Sigma_new_scale);

    // Calculating probab:
    probab = exp( post_sigsq_scaleC(InvSigma_new_scale, prop_lscale, prop_alpha_scale,
                                  covar1, newprop_sigsq_scale, dettnew_scale, sigsq_scale_mean, sigsq_scale_sd) -
                      post_sigsq_scaleC(InvSigma_scale, prop_lscale, prop_alpha_scale,
                                  covar1, prop_sigsq_scale, dett_scale, sigsq_scale_mean, sigsq_scale_sd) );

    // Accept or reject step:
    if (rndm < probab) {
      prop_sigsq_scale = newprop_sigsq_scale;
      Sigma_scale = Sigma_new_scale;
      InvSigma_scale = InvSigma_new_scale;
      dett_scale = dettnew_scale;
      b1 = 1;
    } else {
      newprop_sigsq_scale = prop_sigsq_scale;
      Sigma_new_scale = Sigma_scale;
      InvSigma_new_scale = InvSigma_scale;
      dettnew_scale = dett_scale;
      b1 = 0;
    }

    // Updating the tausq_scale parameter now:
    temp_tausq = rnormscalarC(log(prop_tausq_scale), step_tausq_scale);
    newprop_tausq_scale = exp(temp_tausq);

    // Updating Sigma:
    Sigma_new_scale = get_mat_sigma(dist, prop_beta_scale, prop_sigsq_scale, newprop_tausq_scale);

    InvSigma_new_scale = inv(Sigma_new_scale);
    log_det(dettnew_scale, sign, Sigma_new_scale);

    probab = exp( post_tausq_scaleC(InvSigma_new_scale, prop_lscale, prop_alpha_scale,
                                                          covar1, newprop_tausq_scale, dettnew_scale,
                                                          tausq_scale_mean, tausq_scale_sd) -
                      post_tausq_scaleC(InvSigma_scale, prop_lscale, prop_alpha_scale,
                                                          covar1, prop_tausq_scale, dett_scale,
                                                          tausq_scale_mean, tausq_scale_sd) );

    // Accept or reject step:
    if (rndm < probab) {
      prop_tausq_scale = newprop_tausq_scale;
      Sigma_scale = Sigma_new_scale;
      InvSigma_scale = InvSigma_new_scale;
      dett_scale = dettnew_scale;
      b2 = 1;
    } else {
      newprop_tausq_scale = prop_tausq_scale;
      Sigma_new_scale = Sigma_scale;
      InvSigma_new_scale = InvSigma_scale;
      dettnew_scale = dett_scale;
      b2 = 0;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////// Shape updates now: //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////

    // Updating the shape parameter now:
    newprop_shape = prop_shape;
    for (int l = 0; l < prop_shape.size(); ++l) {
      newprop_shape[l] = rnormscalarC(prop_shape[l], step_shape(0));
      probab = exp( post_shapeC(data, mu, prop_lscale, newprop_shape, prop_alpha_shape,
                                  covar2, InvSigma_shape, l, dett_shape) -
                      post_shapeC(data, mu, prop_lscale, prop_shape, prop_alpha_shape,
                                  covar2, InvSigma_shape, l, dett_shape) );

    //  Accept or reject step:
      if (rndm < probab) {
        prop_shape = newprop_shape;
        tempu[l] = 1;
      } else {
        newprop_shape = prop_shape;
        tempu[l] = 0;
      }
    }

    // Updating the alpha_shape parameter:
    newprop_alpha_shape = prop_alpha_shape;
    for (int l = 0; l < prop_alpha_shape.size(); ++l) {

      newprop_alpha_shape[l] = rnormscalarC(prop_alpha_shape[l], step_alpha_shape[l]);

      probab = exp (post_alpha_shapeC(newprop_alpha_shape, prop_shape, covar2, InvSigma_shape,
                                            alpha_shape_hyper_mean, alpha_shape_hyper_sd, dett_shape) -
                          post_alpha_shapeC(prop_alpha_shape, prop_shape, covar2, InvSigma_shape,
                                            alpha_shape_hyper_mean, alpha_shape_hyper_sd, dett_shape) ) ;

    //  Accept or reject step:
      if (rndm < probab) {
        prop_alpha_shape = newprop_alpha_shape;
        tempv[l] = 1;
      } else {
        newprop_alpha_shape = prop_alpha_shape;
        tempv[l] = 0;
      }
    }

    // Updating the beta_shape parameter:
    nb = prop_beta_shape.n_rows;
    newprop_beta_shape = reshape(prop_beta_shape, nb, nb);

    for(int k=0; k<nb; ++k) {
      for(int p=0; p<=k; ++p) {
        int c = rand() % lenb_shape;
        newprop_beta_shape(k, p) = beta_shape_hyper_prior(c);
        newprop_beta_shape(p, k) = newprop_beta_shape(k, p);

        // Checking that the proposed matrix is positive definite before updating it by MCMC:
        if (pos_def_testC(newprop_beta_shape) == 1) {
          Sigma_new_shape = get_mat_sigma(dist, newprop_beta_shape, prop_sigsq_shape, prop_tausq_shape);
          InvSigma_new_shape = inv(Sigma_new_shape);
          log_det(dettnew_shape, sign, Sigma_new_shape);

          probab = exp (post_beta_shapeC(InvSigma_new_shape, newprop_beta_shape, prop_shape,
                                            prop_alpha_shape, covar2, dettnew_shape) -
                        post_beta_shapeC(InvSigma_shape, prop_beta_shape, prop_shape,
                                            prop_alpha_shape, covar2, dett_shape) );

          // Accept or reject step:
          if (rndm < probab) {
            prop_beta_shape = newprop_beta_shape;
            Sigma_shape = Sigma_new_shape;
            InvSigma_shape = InvSigma_new_shape;
            dett_shape = dettnew_shape;
            temp_acc_beta_shape(k, p) = 1;
            temp_acc_beta_shape(p, k) = 1;
          } else {
            newprop_beta_shape = prop_beta_shape;
            Sigma_new_shape = Sigma_shape;
            InvSigma_new_shape = InvSigma_shape;
            dettnew_shape = dett_shape;
            temp_acc_beta_shape(k, p) = 0;
            temp_acc_beta_shape(p, k) = 0;
          }
        }
      }
    }

    // Updating the sigsq_shape parameter now:
    temp_sigsq = rnormscalarC(log(prop_sigsq_shape), step_sigsq_shape);
    newprop_sigsq_shape = exp(temp_sigsq);

    // Updating Sigma_shape now:
    Sigma_new_shape = get_mat_sigma(dist, prop_beta_shape, newprop_sigsq_shape, prop_tausq_shape);
    InvSigma_new_shape = inv(Sigma_new_shape);
    log_det(dettnew_shape, sign, Sigma_new_shape);

    // Calculating probab:
    probab = exp( post_sigsq_shapeC(InvSigma_new_shape, prop_shape, prop_alpha_shape,
                                  covar2, newprop_sigsq_shape, dettnew_shape,
                                  sigsq_shape_mean, sigsq_shape_sd) -
                      post_sigsq_shapeC(InvSigma_shape, prop_shape, prop_alpha_shape,
                                  covar2, prop_sigsq_shape, dett_shape,
                                  sigsq_shape_mean, sigsq_shape_sd) );

    // Accept or reject step:
    if (rndm < probab) {
      prop_sigsq_shape = newprop_sigsq_shape;
      Sigma_shape = Sigma_new_shape;
      InvSigma_shape = InvSigma_new_shape;
      dett_shape = dettnew_shape;
      b3 = 1;
    } else {
      newprop_sigsq_shape = prop_sigsq_shape;
      Sigma_new_shape = Sigma_shape;
      InvSigma_new_shape = InvSigma_shape;
      dettnew_shape = dett_shape;
      b3 = 0;
    }

    // Updating the tausq_shape parameter now:
    temp_tausq = rnormscalarC(log(prop_tausq_shape), step_tausq_shape);
    newprop_tausq_shape = exp(temp_tausq);

    // Updating Sigma_shape:
    Sigma_new_shape = get_mat_sigma(dist, prop_beta_shape, prop_sigsq_shape, newprop_tausq_shape);
    InvSigma_new_shape = inv(Sigma_new_shape);
    log_det(dettnew_shape, sign, Sigma_new_shape);

    probab = exp( post_tausq_shapeC(InvSigma_new_shape, prop_shape, prop_alpha_shape,
                                  covar2, newprop_tausq_shape, dettnew_shape,
                                  tausq_shape_mean, tausq_shape_sd) -
                      post_tausq_shapeC(InvSigma_shape, prop_shape, prop_alpha_shape,
                                  covar2, prop_tausq_shape, dett_shape,
                                  tausq_shape_mean, tausq_shape_sd) );

    // Accept or reject step:
    if (rndm < probab) {
      prop_tausq_shape = newprop_tausq_shape;
      Sigma_shape = Sigma_new_shape;
      InvSigma_shape = InvSigma_new_shape;
      dett_shape = dettnew_shape;
      b4 = 1;
    } else {
      newprop_tausq_shape = prop_tausq_shape;
      Sigma_new_shape = Sigma_shape;
      InvSigma_new_shape = InvSigma_shape;
      dettnew_shape = dett_shape;
      b4 = 0;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////// As an alternative, where phi=lscale is considered as a constant /////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Updating the log(scale) parameter as a constant now:
//    newprop_lscale = prop_lscale;
//    for (int l = 0; l < prop_lscale.size(); ++l) {
//      newprop_lscale[l] = rnormscalarC(prop_lscale[l], step_scale(0));
//      probab = exp( post_scaleC2(data, mu, newprop_lscale, prop_shape, l) - //+
//                            //dnormscalarC(newprop_shape[l], shape_hyper_mean, shape_hyper_sd) -
//                          post_shapeC2(data, mu, prop_lscale, prop_shape, l) ); //+
//                            //dnormscalarC(as_scalar(prop_shape[l]), shape_hyper_mean, shape_hyper_sd) );
//
//    //  Come back to this:    if(is.na(probab)) {probab <- 0}
//      if (rndm < probab) {
//        prop_lscale = newprop_lscale;
//      } else {
//        newprop_lscale = prop_lscale;
//      }
//    }
//    
//    ///////////////////////////////////////////////////////////////////////////////////////////////
//    /////////////////// As an alternative, where xi=shape is considered as a constant /////////////
//    ///////////////////////////////////////////////////////////////////////////////////////////////
//    // Updating the shape parameter as a constant now:
//    newprop_shape = prop_shape;
//    for (int l = 0; l < prop_shape.size(); ++l) {
//      newprop_shape[l] = rnormscalarC(prop_shape[l], step_shape(0));
//      probab = exp( post_shapeC2(data, mu, prop_lscale, newprop_shape, l) - //+
//                            //dnormscalarC(newprop_shape[l], shape_hyper_mean, shape_hyper_sd) -
//                          post_shapeC2(data, mu, prop_lscale, prop_shape, l) ); //+
//                            //dnormscalarC(as_scalar(prop_shape[l]), shape_hyper_mean, shape_hyper_sd) );
//
//    //  Come back to this:    if(is.na(probab)) {probab <- 0}
//      if (rndm < probab) {
//        prop_shape = newprop_shape;
//      } else {
//        newprop_shape = prop_shape;
//      }
//    }

  //////////////////////////////// Comment out one or other pieces above ////////////////////////

  //////////////////////////////// Now to calculate the acceptance rates: ///////////////////////
  acc_lscale = acc_lscale + tempr;
  acc_alpha_scale = acc_alpha_scale + temps;
  vec temp = as<vec>(temp_acc_beta_scale);
  acc_beta_scale = acc_beta_scale + temp;
  acc_sigsq_scale = acc_sigsq_scale + b1;
  acc_tausq_scale = acc_tausq_scale + b2;

  acc_shape = acc_shape + tempu;
  acc_alpha_shape = acc_alpha_shape + tempv;
  vec temp2 = as<vec>(temp_acc_beta_shape);
  acc_beta_shape = acc_beta_shape + temp2;
  acc_sigsq_shape = acc_sigsq_shape + b3;
  acc_tausq_shape = acc_tausq_shape + b4;

  /////////////////// Saving these at selected iterations: //////////////////////////////////////

  // Save if we're beyond the burn-in period *and* it's every nth iteration:
  if(i > burnin && i % nth == 0) {
    
    // Printing the value of i
    Rprintf("%d \n", i);
    
    m = lscale.n_rows;

    // Writing all the values at each selected cycle for the scale:
    lscale.insert_rows(m, prop_lscale.t());
    alpha_scale.insert_rows(m, prop_alpha_scale.t());
    beta_scale.insert_rows(m, vectorise(prop_beta_scale).t());
    sigsq_scale.push_back(prop_sigsq_scale);
    tausq_scale.push_back(prop_tausq_scale);

    // Writing all the values at each selected cycle for the shape:
    shape.insert_rows(m, prop_shape.t());
    alpha_shape.insert_rows(m, prop_alpha_shape.t());
    beta_shape.insert_rows(m, vectorise(prop_beta_shape).t());
    sigsq_shape.push_back(prop_sigsq_shape);
    tausq_shape.push_back(prop_tausq_shape);
  }
  }

  //////////////////////////// Getting ready to output results: //////////////////////
  // Writing each element of the list:
  out[0] = exp(lscale);
  out[1] = alpha_scale;
  out[2] = beta_scale;
  out[3] = sigsq_scale;
  out[4] = tausq_scale;

  out[5] = shape;
  out[6] = alpha_shape;
  out[7] = beta_shape;
  out[8] = sigsq_shape;
  out[9] = tausq_shape;

  out[10] = acc_lscale/iterations;
  out[11] = acc_alpha_scale/iterations;
  out[12] = acc_beta_scale/iterations; // Because updates are *discrete*, sometimes a 'new' value, the same as the old
                                      // is accepted, leading to slightly inflated acceptance rates
  out[13] = acc_sigsq_scale/iterations;
  out[14] = acc_tausq_scale/iterations;

  out[15] = acc_shape/iterations;
  out[16] = acc_alpha_shape/iterations;
  out[17] = acc_beta_shape/iterations;
  out[18] = acc_sigsq_shape/iterations;
  out[19] = acc_tausq_shape/iterations;

  out[20] = covar1;
  out[21] = covar2;

  // Creating the names for all the elements of the output list:
  int g1 = out.size();
  CharacterVector names(g1);
  names[0] = "scale";
  names[1] = "alpha_scale";
  names[2] = "beta_scale";
  names[3] = "sigsq_scale";
  names[4] = "tausq_scale";

  names[5] = "shape";
  names[6] = "alpha_shape";
  names[7] = "beta_shape";
  names[8] = "sigsq_shape";
  names[9] = "tausq_shape";

  names[10] = "acc_scale";
  names[11] = "acc_alpha_scale";
  names[12] = "acc_beta_scale";
  names[13] = "acc_sigsq_scale";
  names[14] = "acc_tausq_scale";

  names[15] = "acc_shape";
  names[16] = "acc_alpha_shape";
  names[17] = "acc_beta_shape";
  names[18] = "acc_sigsq_shape";
  names[19] = "acc_tausq_shape";

  out.attr("names") = names;

  return out; 
}



