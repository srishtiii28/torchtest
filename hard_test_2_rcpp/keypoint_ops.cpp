// keypoint_ops.cpp
// Hard Test 2 – Rcpp Interface: exposing C++ functions to R
//
// Two functions are exported:
//   1. nms_cpp()                 – Non-Maximum Suppression for bounding boxes
//   2. keypoint_distances_cpp()  – Pairwise Euclidean distances between keypoints
//
// Both are directly relevant to the torchvision object-detection pipeline.
//
// Standalone usage:  Rcpp::sourceCpp("hard_test_2_rcpp/keypoint_ops.cpp")
// In a package:      place in src/, run devtools::document() / R CMD INSTALL

// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace Rcpp;

// ═══════════════════════════════════════════════════════════════════════════
// 1.  Non-Maximum Suppression (NMS)
// ═══════════════════════════════════════════════════════════════════════════

// Helper: IoU between two rows of a box matrix (x1,y1,x2,y2)
static double iou(const NumericMatrix& boxes, int i, int j) {
  double x1 = std::max(boxes(i, 0), boxes(j, 0));
  double y1 = std::max(boxes(i, 1), boxes(j, 1));
  double x2 = std::min(boxes(i, 2), boxes(j, 2));
  double y2 = std::min(boxes(i, 3), boxes(j, 3));

  double inter_w = std::max(0.0, x2 - x1);
  double inter_h = std::max(0.0, y2 - y1);
  double inter   = inter_w * inter_h;
  if (inter == 0.0) return 0.0;

  double area_i = (boxes(i, 2) - boxes(i, 0)) * (boxes(i, 3) - boxes(i, 1));
  double area_j = (boxes(j, 2) - boxes(j, 0)) * (boxes(j, 3) - boxes(j, 1));

  return inter / (area_i + area_j - inter);
}

//' Non-Maximum Suppression (C++)
//'
//' Greedily selects bounding boxes in order of decreasing score, suppressing
//' boxes whose IoU with a selected box exceeds \code{iou_threshold}.
//' This is the core post-processing step used by Faster R-CNN and YOLO.
//'
//' @param boxes         Numeric matrix (N x 4): columns are x1, y1, x2, y2.
//' @param scores        Numeric vector of length N with detection confidence.
//' @param iou_threshold Overlap threshold above which a box is suppressed (default 0.5).
//'
//' @return Integer vector of 1-based indices of kept boxes, sorted by score.
//'
//' @examples
//' boxes <- matrix(c(
//'   10, 10, 50, 50,
//'   12, 12, 52, 52,
//'   100, 100, 150, 150
//' ), ncol = 4, byrow = TRUE)
//' scores <- c(0.9, 0.8, 0.95)
//' nms_cpp(boxes, scores)
//' # Returns c(3, 1)  – box 2 suppressed (IoU > 0.5 with box 1)
//'
//' @export
// [[Rcpp::export]]
IntegerVector nms_cpp(NumericMatrix boxes,
                      NumericVector scores,
                      double iou_threshold = 0.5) {

  int n = boxes.nrow();
  if (n == 0) return IntegerVector(0);
  if (scores.size() != n)
    stop("`scores` length must equal number of rows in `boxes`.");

  // Sort by descending score
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b){ return scores[a] > scores[b]; });

  std::vector<bool> suppressed(n, false);
  std::vector<int>  keep;

  for (int ii = 0; ii < n; ++ii) {
    int i = order[ii];
    if (suppressed[i]) continue;
    keep.push_back(i + 1);   // 1-based for R
    for (int jj = ii + 1; jj < n; ++jj) {
      int j = order[jj];
      if (!suppressed[j] && iou(boxes, i, j) > iou_threshold)
        suppressed[j] = true;
    }
  }

  return wrap(keep);
}


// ═══════════════════════════════════════════════════════════════════════════
// 2.  Pairwise keypoint distances
// ═══════════════════════════════════════════════════════════════════════════

//' Pairwise Euclidean Distances Between Keypoints (C++)
//'
//' Computes the full symmetric K×K distance matrix for a set of K keypoints.
//' Useful for pose geometry analysis (limb length, skeleton symmetry).
//'
//' @param keypoints Numeric matrix (K x 2) with columns x, y.
//'
//' @return A K×K symmetric NumericMatrix; entry [i,j] is the Euclidean
//'   distance between keypoint i and keypoint j.  Diagonal is 0.
//'
//' @examples
//' kpts <- matrix(c(0, 0,
//'                  3, 4,
//'                  6, 0), ncol = 2, byrow = TRUE)
//' keypoint_distances_cpp(kpts)
//' # row 1→2: dist = 5  (classic 3-4-5 triangle)
//'
//' @export
// [[Rcpp::export]]
NumericMatrix keypoint_distances_cpp(NumericMatrix keypoints) {
  int k = keypoints.nrow();
  NumericMatrix dist(k, k);

  for (int i = 0; i < k; ++i) {
    for (int j = i + 1; j < k; ++j) {
      double dx = keypoints(i, 0) - keypoints(j, 0);
      double dy = keypoints(i, 1) - keypoints(j, 1);
      double d  = std::sqrt(dx * dx + dy * dy);
      dist(i, j) = d;
      dist(j, i) = d;
    }
  }

  if (!Rf_isNull(rownames(keypoints))) {
    rownames(dist) = rownames(keypoints);
    colnames(dist) = rownames(keypoints);
  }

  return dist;
}
