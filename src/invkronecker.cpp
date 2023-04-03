#include <Rcpp.h>
#include <RcppEigen.h>

//' This function computes a matrix operation for the computation of the predM1
//' matrix. 
//' @export
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::MatrixXd compute_predM1_rcpp(const Eigen::Map<Eigen::MatrixXd> cy,
                                    const Eigen::Map<Eigen::MatrixXd> k,
                                    const Eigen::Map<Eigen::MatrixXd> e,
                                    const Eigen::Map<Eigen::MatrixXd> ie) {
	
	// kronecker product
	Eigen::MatrixXd kr_cy_k(cy.rows()*k.rows(), cy.cols()*k.cols());
	Eigen::MatrixXd kr_e_ie(e.rows()*ie.rows(), e.cols()*ie.cols());
	
	for (int i = 0; i < cy.cols(); i++) {
	  for (int j = 0; j < cy.rows(); j++) {
		kr_cy_k.block(i*k.rows(), j*k.cols(), k.rows(), k.cols()) =  cy(i,j)*k;
	  }
	}
	
		for (int i = 0; i < e.cols(); i++) {
	  for (int j = 0; j < e.rows(); j++) {
		kr_e_ie.block(i*ie.rows(), j*ie.cols(), ie.rows(), ie.cols()) =  e(i,j)*ie;
	  }
	}
	
	Eigen::MatrixXd kr_sum = kr_cy_k + kr_e_ie;
	
	// Matrix inverse from Cholesky decomposition
	Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> lltofkr_sum(kr_sum);
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(kr_sum.rows(), kr_sum.cols());
	Eigen::MatrixXd inv_chol = lltofkr_sum.solve(I);
	
  return inv_chol;
}
