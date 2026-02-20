# draw_keypoints() – Fixed and Complete Implementation
# For: GSOC 2026 Hard Test 1 (Visualization)
#
# Bugs found in the original (mlverse/torchvision main branch):
#   BUG 1: `for (pose in dim(img_kpts)[[1]])` iterates over a single integer,
#           not 1:N.  Must be `seq_len(dim(img_kpts)[[1]])`.
#   BUG 2: Double division by 255.  img_to_draw is already normalised to [0,1]
#           by `image$div(255)`, then `png::writePNG(img_to_draw / 255)` divides
#           again, producing an almost-black image.
#   BUG 3: `colors = NULL` never triggers viridis generation – the loop just
#           passes NULL to graphics::points, which silently uses the default.
#   BUG 4: `connectivity` parameter is accepted but completely ignored.
#   BUG 5: `width` parameter is accepted but never used.
#   BUG 6: `graphics::points(..., pch = ".")` draws a tiny dot regardless of
#           `radius`; cex is not in pixels.  Use graphics::symbols() for
#           pixel-accurate filled circles.

#' Draws Keypoints
#'
#' Draws keypoints, objects describing body parts (e.g. rightArm,
#' leftShoulder), on a given RGB tensor image.
#'
#' @param image   Tensor of shape (3 x H x W) and dtype \code{uint8} or
#'   \code{float}.  For float images values are assumed to be in \eqn{[0,1]}.
#' @param keypoints Tensor of shape (N, K, 2): K keypoint locations for each
#'   of the N detected pose instances, in \code{c(x, y)} format.
#' @param connectivity List of integer pairs \code{list(c(i,j), ...)} naming
#'   keypoint indices to connect with a line.  \code{NULL} (default) draws no
#'   connecting lines.
#' @param colors Character vector of colours for each pose instance, or a
#'   single colour applied to all.  Accepts R colour names or hex strings
#'   (\code{"red"}, \code{"#FF00FF"}).  When \code{NULL} (default), colours
#'   are drawn from a rainbow palette.
#' @param radius Radius in pixels of each plotted keypoint circle.  Default 2.
#' @param width  Line width (in pixels) for connectivity lines.  Default 3.
#'
#' @return A tensor of dtype \code{uint8} and shape (3, H, W) with keypoints
#'   (and optional connecting lines) rendered onto the image.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#'   image     <- torch::torch_randint(190L, 255L, size = c(3L, 360L, 360L),
#'                                     dtype = torch::torch_uint8())
#'   keypoints <- torch::torch_randint(60L, 300L, size = c(4L, 5L, 2L))
#'
#'   # Skeleton connectivity for a 5-keypoint body model
#'   connectivity <- list(c(1, 2), c(2, 3), c(3, 4), c(4, 5))
#'
#'   result <- draw_keypoints(image, keypoints,
#'                            connectivity = connectivity,
#'                            colors = c("red", "blue", "green", "orange"),
#'                            radius = 5, width = 2)
#'   torchvision::tensor_image_browse(result)
#' }
#'
#' @family image display
#' @export
draw_keypoints <- function(image,
                           keypoints,
                           connectivity = NULL,
                           colors = NULL,
                           radius = 2,
                           width = 3) {

  rlang::check_installed("magick")
  rlang::check_installed("png")

  # ── Input validation ────────────────────────────────────────────────────────
  if (!inherits(image, "torch_tensor"))
    rlang::abort("`image` must be a torch_tensor.")
  if (image$ndim != 3L)
    rlang::abort("Pass an individual `image`, not a batch (ndim must be 3).")
  if (!image$size(1) %in% c(1L, 3L))
    rlang::abort("Only grayscale (C=1) and RGB (C=3) images are supported.")
  if (keypoints$ndim != 3L)
    rlang::abort(sprintf(
      "`keypoints` must be of shape (N, K, 2); got shape (%s).",
      paste(keypoints$shape, collapse = ", ")))
  if (keypoints$size(3) != 2L)
    rlang::abort("`keypoints` last dimension must be 2 (x, y).")

  # ── Convert image tensor → [0,1] numeric array (H x W x C) ─────────────────
  # FIX BUG 2: only one division by 255, right here.
  if (image$dtype == torch::torch_uint8()) {
    img_array <- image$div(255)$permute(c(2L, 3L, 1L))$to(device = "cpu") |>
      as.array()
  } else if (image$dtype == torch::torch_float()) {
    img_array <- image$permute(c(2L, 3L, 1L))$to(device = "cpu") |>
      as.array()
  } else {
    rlang::abort("`image` dtype must be torch_uint8 or torch_float.")
  }

  num_instances <- keypoints$shape[[1]]
  num_kpts      <- keypoints$shape[[2]]

  # ── Colour palette ──────────────────────────────────────────────────────────
  # FIX BUG 3: actually generate colours when NULL.
  if (is.null(colors)) {
    colors <- grDevices::rainbow(num_instances, alpha = 1)
  } else if (length(colors) == 1L) {
    colors <- rep(colors, num_instances)
  }

  # ── Convert keypoints to integer array ─────────────────────────────────────
  img_kpts <- keypoints$to(torch::torch_int64())$to(device = "cpu") |>
    as.array()   # shape: [N, K, 2]

  # ── Open magick drawing canvas ──────────────────────────────────────────────
  # FIX BUG 2 (cont.): img_array is already [0,1]; no second /255.
  draw <- png::writePNG(img_array) |>
    magick::image_read() |>
    magick::image_draw()

  # ── Draw each pose instance ─────────────────────────────────────────────────
  # FIX BUG 1: seq_len() so we iterate 1, 2, …, N (not just the value N).
  for (pose in seq_len(num_instances)) {
    col   <- colors[[pose]]
    kpt_x <- img_kpts[pose, , 1]
    kpt_y <- img_kpts[pose, , 2]

    # FIX BUG 6: use symbols() so `radius` is in pixel units.
    graphics::symbols(
      x       = kpt_x,
      y       = kpt_y,
      circles = rep(radius, num_kpts),
      inches  = FALSE,
      add     = TRUE,
      fg      = col,
      bg      = col
    )

    # FIX BUG 4 & 5: implement connectivity + use `width`.
    if (!is.null(connectivity)) {
      for (pair in connectivity) {
        i <- pair[[1]]
        j <- pair[[2]]
        graphics::lines(
          x   = c(kpt_x[[i]], kpt_x[[j]]),
          y   = c(kpt_y[[i]], kpt_y[[j]]),
          col = col,
          lwd = width
        )
      }
    }
  }

  grDevices::dev.off()

  # ── Convert drawn canvas back to uint8 tensor (3 x H x W) ──────────────────
  draw_tt <- draw |>
    magick::image_data(channels = "rgb") |>
    as.integer() |>
    torch::torch_tensor(dtype = torch::torch_uint8())

  draw_tt$permute(c(3L, 1L, 2L))
}
