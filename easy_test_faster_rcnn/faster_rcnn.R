# Loading Images ---------------------------------------------------
library(torchvision)
library(torch)

url1 <- "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"
url2 <- "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg"

dog1 <- magick_loader(url1) %>% transform_to_tensor()
dog2 <- magick_loader(url2) %>% transform_to_tensor()


# Visualizing a grid of images -------------------------------------


dogs <- torch_stack(list(dog1, dog2))
grid <- vision_make_grid(dogs, scale = TRUE, num_rows = 2)
tensor_image_browse(grid)


# Preprocessing the data -------------------------------------------
# faster_rcnn handles its own internal resizing; we only need float [0,1] tensors.
# Images are stacked into a 4D batch tensor (N, C, H, W).

dog_batch <- torch_stack(list(dog1, dog2))


# Loading Model ----------------------------------------------------
# model_fasterrcnn_resnet50_fpn_v2 is the latest evolution of the
# Faster R-CNN family: improved ResNet-50 FPN v2 backbone with updated
# training recipes, delivering higher mAP than the original v1.

model <- model_fasterrcnn_resnet50_fpn_v2(pretrained = TRUE,
                                          score_thresh = 0.01,
                                          nms_thresh   = 0.5)
model$eval()

output <- model(dog_batch)


# Processing the Output --------------------------------------------
# The model returns a named list:
#   $detections – list of per-image results, each with:
#     $boxes   float tensor (N, 4)  bounding boxes (x1, y1, x2, y2)
#     $labels  int64 tensor (N,)    COCO class indices
#     $scores  float tensor (N,)    confidence scores

det1 <- output$detections[[1]]
det2 <- output$detections[[2]]

cat("Image 1 –", as.integer(det1$boxes$size(1)), "detections\n")
cat("Image 2 –", as.integer(det2$boxes$size(1)), "detections\n")

# Keep only the top-5 highest-scoring boxes per image
top_k <- function(det, k = 5) {
  n <- as.integer(det$boxes$size(1))
  if (n == 0) return(det)
  idx <- seq_len(min(k, n))
  list(
    boxes  = det$boxes[idx, ],
    labels = det$labels[idx],
    scores = det$scores[idx]
  )
}

top1 <- top_k(det1)
top2 <- top_k(det2)


# Decode COCO labels (requires network; falls back to integer indices) ------
safe_labels <- function(label_tensor) {
  ids <- as.integer(label_tensor)
  tryCatch(
    coco_label(ids),
    error = function(e) paste0("class_", ids)
  )
}

# Filter geometrically invalid boxes (x1 >= x2 or y1 >= y2)
valid_boxes <- function(det) {
  n <- as.integer(det$boxes$size(1))
  if (n == 0) return(det)
  b    <- as.matrix(det$boxes$to(device = "cpu") |> as.array())
  keep <- which((b[, 3] > b[, 1]) & (b[, 4] > b[, 2]))
  list(
    boxes  = det$boxes[keep, , drop = FALSE],
    labels = det$labels[keep],
    scores = det$scores[keep]
  )
}


# Visualizing the Output -------------------------------------------


draw_detections <- function(image, det) {
  det <- valid_boxes(det)
  n   <- length(as.integer(det$labels))
  if (n == 0) return((image * 255)$to(dtype = torch_uint8()))
  draw_bounding_boxes(
    (image * 255)$to(dtype = torch_uint8()),
    boxes  = det$boxes$view(c(-1L, 4L)),
    labels = safe_labels(det$labels)
  )
}

result1 <- draw_detections(dog1, top1)
result2 <- draw_detections(dog2, top2)

tensor_image_browse(result1)
tensor_image_browse(result2)


# Comparing v1 vs v2 (latest evolution) ---------------------------


model_v1 <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE,
                                          score_thresh = 0.01,
                                          nms_thresh   = 0.5)
model_v1$eval()
out_v1 <- model_v1(dog_batch)

cat("v1 detections on dog1:", as.integer(out_v1$detections[[1]]$boxes$size(1)), "\n")
cat("v2 detections on dog1:", as.integer(det1$boxes$size(1)), "\n")
