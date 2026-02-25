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


# Preprocessing the data -------------------------------------


# faster_rcnn expects float tensors normalised to [0, 1]
dog1_prep <- dog1 %>% transform_to_tensor()
dog2_prep <- dog2 %>% transform_to_tensor()

# The model accepts a plain list of individual image tensors (no batching needed)
dog_batch <- list(dog1_prep, dog2_prep)


# Loading Model -------------------------------------


# model_fasterrcnn_resnet50_fpn_v2 is the latest evolution of the Faster R-CNN
# family in torchvision: it uses an improved ResNet-50 FPN v2 backbone with
# updated training recipes, delivering higher mAP than the original v1 model.
model <- model_fasterrcnn_resnet50_fpn_v2(pretrained = TRUE,
                                          score_thresh = 0.5,
                                          nms_thresh   = 0.5)
model$eval()

# run model
preds <- model(dog_batch)


# Processing the Output ------------------------------

# Each element of `preds` is a named list for one image:
#   $boxes  – float tensor (N, 4)  bounding boxes in (x1, y1, x2, y2) format
#   $labels – int64 tensor (N,)    COCO class indices (1-indexed)
#   $scores – float tensor (N,)    confidence scores in [0, 1]
pred1 <- preds[[1]]
pred2 <- preds[[2]]

pred1$boxes$shape   # (N, 4)
pred1$labels        # COCO label indices
pred1$scores        # confidence scores


# Visualizing the Output ------------------------------


# Helper to draw boxes on a single image
draw_detections <- function(image, pred) {
  num_boxes <- as.integer(pred$boxes$size(1))
  if (num_boxes == 0) return(image)

  boxes  <- pred$boxes$view(c(-1, 4))
  labels <- coco_label(as.integer(pred$labels))

  draw_bounding_boxes(
    (image * 255)$to(dtype = torch_uint8()),
    boxes  = boxes,
    labels = labels
  )
}

result1 <- draw_detections(dog1_prep, pred1)
result2 <- draw_detections(dog2_prep, pred2)

tensor_image_browse(result1)
tensor_image_browse(result2)


# Comparing v1 vs v2 (latest evolution) ---------------------------


model_v1 <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE,
                                          score_thresh = 0.5,
                                          nms_thresh   = 0.5)
model_v1$eval()
preds_v1 <- model_v1(dog_batch)

cat("v1 detections on dog1:", as.integer(preds_v1[[1]]$boxes$size(1)), "\n")
cat("v2 detections on dog1:", as.integer(pred1$boxes$size(1)), "\n")
