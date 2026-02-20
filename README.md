# torchtest – GSOC 2026 Qualifying Tests

Hard qualifying tests for the [torchvision C++ Performance Improvements](https://github.com/rstats-gsoc/gsoc2026/wiki/torchvision-and-ecosystem) project.

---

## Hard Test 1 – Complete `draw_keypoints()`

**Directory:** `hard_test_1_draw_keypoints/`

### Bugs found in `mlverse/torchvision` main branch

| # | Bug | Fix |
|---|-----|-----|
| 1 | `for (pose in dim(img_kpts)[[1]])` — iterates over the single value N, not `1:N` | `seq_len(num_instances)` |
| 2 | `png::writePNG(img_to_draw / 255)` — image already divided by 255, second division makes the output near-black | removed the second `/ 255` |
| 3 | `colors = NULL` silently passes NULL to `graphics::points()`, no palette generated | `grDevices::rainbow(num_instances)` fallback |
| 4 | `connectivity` parameter accepted but never used | added `graphics::lines()` loop over pairs |
| 5 | `width` parameter accepted but never used | passed as `lwd = width` to `graphics::lines()` |
| 6 | `graphics::points(pch = ".")` — `cex` scales a character glyph, not pixels | switched to `graphics::symbols(inches = FALSE)` for pixel-accurate circles |

### Run

```r
library(torch)
source("hard_test_1_draw_keypoints/draw_keypoints_demo.R")
```

---

## Hard Test 2 – Rcpp Interface

**Directory:** `hard_test_2_rcpp/`

Two C++ functions exposed to R via `// [[Rcpp::export]]`:

| Function | Description |
|----------|-------------|
| `nms_cpp(boxes, scores, iou_threshold)` | Non-Maximum Suppression — same algorithm used inside Faster R-CNN |
| `keypoint_distances_cpp(keypoints)` | Pairwise K×K Euclidean distance matrix for keypoints |

### Run

```r
install.packages("Rcpp")   # once
source("hard_test_2_rcpp/demo.R")
```
