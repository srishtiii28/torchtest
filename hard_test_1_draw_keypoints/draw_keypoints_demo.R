# draw_keypoints() – Demo Script
# Run this after sourcing draw_keypoints_fixed.R.
# Requires: torch, magick, png

library(torch)
library(magick)

source("hard_test_1_draw_keypoints/draw_keypoints_fixed.R")

# Helper: display a uint8 tensor as an image
show_tensor <- function(t, title = "") {
  arr <- t$permute(c(2L, 3L, 1L))$to(device = "cpu") |> as.array()
  img <- magick::image_read(png::writePNG(arr / 255))
  if (nchar(title)) message(title)
  print(img)
  invisible(img)
}

# ════════════════════════════════════════════════════════════════════════════
# Demo 1 – Basic keypoints, auto rainbow colours (no connectivity)
# ════════════════════════════════════════════════════════════════════════════
message("Demo 1: basic keypoints, auto rainbow colours")

set.seed(42)
image <- torch_randint(180L, 230L, size = c(3L, 400L, 400L),
                       dtype = torch_uint8())
# 3 persons, each with 5 keypoints
keypoints <- torch_randint(50L, 350L, size = c(3L, 5L, 2L))

result1 <- draw_keypoints(image, keypoints, radius = 6)
show_tensor(result1, "Demo 1 – basic keypoints")

stopifnot(result1$dtype == torch_uint8())
stopifnot(result1$shape == c(3L, 400L, 400L))

# ════════════════════════════════════════════════════════════════════════════
# Demo 2 – Skeleton connectivity lines + custom colours
# ════════════════════════════════════════════════════════════════════════════
message("Demo 2: connectivity skeleton + custom colours")

# 5-keypoint skeleton: head, l-shoulder, r-shoulder, l-hip, r-hip
connectivity <- list(
  c(1L, 2L),   # head → left shoulder
  c(1L, 3L),   # head → right shoulder
  c(2L, 4L),   # left shoulder → left hip
  c(3L, 5L),   # right shoulder → right hip
  c(4L, 5L)    # left hip → right hip
)

result2 <- draw_keypoints(
  image,
  keypoints,
  connectivity = connectivity,
  colors       = c("#E74C3C", "#3498DB", "#2ECC71"),
  radius       = 8,
  width        = 2
)
show_tensor(result2, "Demo 2 – skeleton connectivity")

# ════════════════════════════════════════════════════════════════════════════
# Demo 3 – Float image input
# ════════════════════════════════════════════════════════════════════════════
message("Demo 3: float32 image input")

image_float <- image$to(dtype = torch_float())$div(255)
result3 <- draw_keypoints(image_float, keypoints,
                          connectivity = connectivity,
                          radius = 5, width = 2)
show_tensor(result3, "Demo 3 – float image")

stopifnot(result3$dtype == torch_uint8())

message("All assertions passed.")
