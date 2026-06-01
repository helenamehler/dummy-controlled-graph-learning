####################################################################################################################
############ Calculate fdp and power based on the estimated edge set(s) E and true precision matrix Omega #########

#######################
##### For one estimated E
Fdp_Power_Graph_func <- function(E_est, Omega){
  eps <- .Machine$double.eps
  if (sum(abs(diag(E_est))) != 0) print("Diagonals of E_est are not 0!")
  
  ## true adjacency (off-diagonal, both (i,j) and (j,i) counted)
  diag(Omega) <- 0
  adj.true <- (abs(Omega) > eps) + 0
  num.edge <- sum(adj.true > eps)            # true edges (doubled)
  
  ## binarize estimate
  E_est <- (abs(E_est) > eps) + 0
  diag(E_est) <- 0
  
  num.dis <- sum(E_est > eps)                # discoveries (doubled)
  num.fd  <- sum((adj.true - E_est) < -eps)  # false discoveries (doubled)
  num.td  <- sum(2 * adj.true - E_est == 1)  # true discoveries (doubled)
  
  pp <- nrow(adj.true)
  num.nonedge <- pp * (pp - 1) - num.edge    # true non-edges (doubled)
  
  fdp      <- num.fd / max(num.dis, 1)
  power    <- if (num.edge == 0) 0 else num.td / num.edge
  fpr      <- if (num.nonedge == 0) 0 else num.fd / num.nonedge
  fp_edges <- num.fd / 2                      # unique false edges (i<j)
  
  return(c(fdp = fdp, power = power, fpr = fpr, fp_edges = fp_edges))
}

#######################
##### For a list of estimated E
Fdp_Power_GraphList_func <- function(E_est_list, Omega){

  num_E <- length(E_est_list)

  fdp_power_index_func <- function(index,E_est_list, Omega){
    E_est <- E_est_list[[index]]
    return(Fdp_Power_Graph_func(E_est, Omega))
  }

  fdr_power_all_algo <- t(mapply(fdp_power_index_func, 1:num_E, MoreArgs=list(E_est_list, Omega)))
  return(fdr_power_all_algo)
}
