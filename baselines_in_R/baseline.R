library(ggcorrplot)
library(mvtnorm)
library(SILGGM)
library(igraph)


# Load source files
source_files <- c(
  "other/Graphs.R", "other/BHBY.R", "other/FdrPowerGraphFunc.R")
lapply(source_files, source)

# Parameters
p <- 150
prob <- 5 / p
range <- c(0.2, 0.8)
add_minEigen <- 0.15
signs <- "Random"
alpha <- seq(0.05, 1, by = 0.05)
eps <- .Machine$double.eps
B <- 100  # MC runs
n <- 151
raw_results <- data.frame(
  n = integer(),
  alpha = numeric(),
  fdp = numeric(),
  tpp = numeric(),
  fpr = numeric(),  
  fp_edges = integer(),     # neu: false positive (unique edges, i<j)
  mode = character(),
  stringsAsFactors = FALSE
)



nei = 5
power = 0.5
m = 5


for (a in alpha) {
  cat("Running for n =", n, "\n")
  
  for (b in 1:B) {
    set.seed(1000 * n + b)
    
    # New graph each time
    #graph_result <- Small_World_graph(p, prob, range, nei, add_minEigen, signs)
    graph_result <- ER_graph(p, prob, range, add_minEigen, signs)
    #graph_result <- PA_graph(p, range, power, m, add_minEigen, signs)
    Omega <- graph_result[[1]]
    Sigma <- graph_result[[2]]
    mu <- rep(0, p)
    
    # New data
    X <- rmvnorm(n, mu, Sigma)

    
    # BH
    A_BH <- Multi_Test_Graph_func(X, a, method = "BH")
    fdpp <- Fdp_Power_Graph_func(A_BH, Omega)
    raw_results <- rbind(raw_results, data.frame(
      n = n,
      alpha = a,
      fdp = fdpp["fdp"],
      tpp = fdpp["power"],
      fpr = fdpp["fpr"],
      fp_edges = as.integer(fdpp["fp_edges"]),
      mode = "BH"
    ))
    
    
    out_GFC_L <- SILGGM(X, method="GFC_L", alpha = a, true_graph = Omega)
    
    A_Liu <- out_GFC_L$global_decision[[1]]
    A_Liu <- ((A_Liu == 1) | (t(A_Liu) == 1)) * 1
    diag(A_Liu) <- 0
    
    fdpp <- Fdp_Power_Graph_func(A_Liu, Omega)
    
    raw_results <- rbind(raw_results, data.frame(
      n = n,
      alpha = a,
      fdp = fdpp["fdp"],
      tpp = fdpp["power"],
      fpr = fdpp["fpr"],
      fp_edges = as.integer(fdpp["fp_edges"]),
      mode = "Liu"
    ))
    
    
    
    if (b %% 10 == 0) cat("  Finished run", b, "of", B, "\n")
  }
}

agg_results <- aggregate(
  cbind(fdp, tpp, fpr, fp_edges) ~ alpha + mode,
  data = raw_results,
  FUN = function(x) mean(x, na.rm = TRUE)
)

colnames(agg_results) <- c("alpha", "mode", "fdp_mean", "tpp_mean", "fpr_mean", "fp_mean")

modes <- unique(agg_results$mode)


agg_results <- agg_results[order(agg_results$mode, agg_results$alpha), ]

print(agg_results)

write.csv(agg_results,
          file = "r_results_sim12_er.csv",
          row.names = FALSE)