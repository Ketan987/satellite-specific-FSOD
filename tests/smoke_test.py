"""Simple smoke test for FSOD model forward/predict using synthetic data"""

import torch
from models.detector import FSODDetector


def run_smoke_test():
    # Small image size to keep feature maps tiny
    image_size = 128

    # Instantiate model without pretrained weights for speed
    model = FSODDetector(feature_dim=2048, embed_dim=256, image_size=image_size, pretrained=False)
    model.eval()

    # Synthetic support: 3 support images
    num_support = 3
    support_images = torch.randn(num_support, 3, image_size, image_size)

    # One bbox per support image: [x, y, w, h]
    support_boxes = [torch.tensor([[10.0, 10.0, 30.0, 30.0]]) for _ in range(num_support)]

    # Synthetic query image
    query_image = torch.randn(1, 3, image_size, image_size)

    # Run predict (use CPU to avoid GPU issues)
    with torch.no_grad():
        # simple support labels (0,1,2)
        support_labels = torch.tensor([0, 1, 2], dtype=torch.long)
        boxes, scores, preds = model.predict(support_images, support_boxes, support_labels, query_image, score_threshold=0.0, nms_threshold=1.0, max_detections=10, n_way=3)

    print("Smoke test passed. Returned boxes:", boxes.shape, "scores:", scores.shape)


if __name__ == '__main__':
    run_smoke_test()
