# Hard Test 2 – Rcpp Interface Demo
# Requires: Rcpp  (install.packages("Rcpp"))

library(Rcpp)

Rcpp::sourceCpp("hard_test_2_rcpp/keypoint_ops.cpp")
cat("C++ functions loaded via Rcpp::sourceCpp().\n\n")

# ════════════════════════════════════════════════════════════════════════════
# Demo 1 – Non-Maximum Suppression
# ════════════════════════════════════════════════════════════════════════════
cat("── Demo 1: Non-Maximum Suppression (NMS) ──────────────────────────────\n")

boxes <- matrix(c(
  10,  10,  60,  60,   # box 1 – score 0.90
  12,  12,  62,  62,   # box 2 – score 0.75  (overlaps box 1 → suppressed)
  15,  15,  55,  55,   # box 3 – score 0.60  (overlaps box 1 → suppressed)
  200, 200, 250, 250,  # box 4 – score 0.95  (isolated, kept)
  202, 202, 252, 252,  # box 5 – score 0.85  (overlaps box 4 → suppressed)
  400, 100, 480, 180   # box 6 – score 0.70  (isolated, kept)
), ncol = 4, byrow = TRUE)

scores <- c(0.90, 0.75, 0.60, 0.95, 0.85, 0.70)

kept <- nms_cpp(boxes, scores, iou_threshold = 0.5)
cat("Kept indices (1-based):", kept, "\n")
cat("Expected: 4 1 6  (boxes 2, 3, 5 suppressed)\n\n")
stopifnot(setequal(kept, c(1L, 4L, 6L)))

# ════════════════════════════════════════════════════════════════════════════
# Demo 2 – Pairwise keypoint distances
# ════════════════════════════════════════════════════════════════════════════
cat("── Demo 2: Pairwise Keypoint Distances ────────────────────────────────\n")

kpts <- matrix(c(
    0,  0,   # head
    0, 80,   # left shoulder  (dist to head = 80)
   60, 80,   # right shoulder (dist to l_shoulder = 60, to head = 100)
   30, 160   # hip
), ncol = 2, byrow = TRUE)
rownames(kpts) <- c("head", "l_shoulder", "r_shoulder", "hip")

D <- keypoint_distances_cpp(kpts)
print(round(D, 2))

stopifnot(abs(D["head", "l_shoulder"]       -  80) < 1e-9)
stopifnot(abs(D["l_shoulder", "r_shoulder"] -  60) < 1e-9)
stopifnot(abs(D["head", "r_shoulder"]       - 100) < 1e-9)
stopifnot(all(diag(D) == 0))
stopifnot(all(D == t(D)))
cat("Distance checks passed.\n\n")

# ════════════════════════════════════════════════════════════════════════════
# Demo 3 – NMS benchmark (1000 boxes)
# ════════════════════════════════════════════════════════════════════════════
cat("── Demo 3: NMS benchmark – 1 000 random boxes ─────────────────────────\n")
set.seed(42)
N  <- 1000L
x1 <- runif(N, 0, 500); y1 <- runif(N, 0, 500)
x2 <- x1 + runif(N, 10, 100); y2 <- y1 + runif(N, 10, 100)
big_boxes  <- cbind(x1, y1, x2, y2)
big_scores <- runif(N)

t0      <- proc.time()
kept    <- nms_cpp(big_boxes, big_scores)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("NMS on %d boxes: kept %d  |  elapsed %.4f s\n", N, length(kept), elapsed))

cat("\nAll assertions passed. Rcpp interface demo complete.\n")
