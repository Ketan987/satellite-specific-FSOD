"""
Example usage of FSOD inference
"""

import torch
from inference import FSODInference
import json


def example_single_detection():
    """Example: Single query image detection"""
    
    # Initialize inference
    inferencer = FSODInference(
        model_path='checkpoints/best_model.pth',
        device='cuda'
    )
    
    # Define support set (5-shot example for 'cat' class)
    support_set = [
        {
            'image_path': 'data/support/cat1.jpg',
            'bbox': [50, 60, 120, 150],  # [x, y, width, height]
            'class_name': 'cat'
        },
        {
            'image_path': 'data/support/cat2.jpg',
            'bbox': [30, 40, 110, 140],
            'class_name': 'cat'
        },
        {
            'image_path': 'data/support/cat3.jpg',
            'bbox': [70, 80, 130, 160],
            'class_name': 'cat'
        },
        {
            'image_path': 'data/support/cat4.jpg',
            'bbox': [45, 55, 115, 145],
            'class_name': 'cat'
        },
        {
            'image_path': 'data/support/cat5.jpg',
            'bbox': [60, 70, 125, 155],
            'class_name': 'cat'
        }
    ]
    
    # Run detection
    detections = inferencer.detect(
        support_set=support_set,
        query_image='data/query/test_image.jpg',
        score_threshold=0.3,
        nms_threshold=0.4,
        max_detections=100
    )
    
    # Print results
    print(f"Detected {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"\nDetection {i+1}:")
        print(f"  BBox: {det['bbox']}")
        print(f"  Similarity Score: {det['similarity_score']:.4f}")
        print(f"  Class: {det['class_name']}")
    
    # Save results
    output = {
        'query_image': 'data/query/test_image.jpg',
        'detections': detections
    }
    
    with open('detection_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to detection_results.json")


def example_multi_class_detection():
    """Example: Multi-class detection (N-way)"""
    
    inferencer = FSODInference(
        model_path='checkpoints/best_model.pth',
        device='cuda'
    )
    
    # Support set with multiple classes
    support_set = [
        # Cat examples (K-shot)
        {'image_path': 'data/support/cat1.jpg', 'bbox': [50, 60, 120, 150], 'class_name': 'cat'},
        {'image_path': 'data/support/cat2.jpg', 'bbox': [30, 40, 110, 140], 'class_name': 'cat'},
        
        # Dog examples (K-shot)
        {'image_path': 'data/support/dog1.jpg', 'bbox': [40, 50, 130, 160], 'class_name': 'dog'},
        {'image_path': 'data/support/dog2.jpg', 'bbox': [55, 65, 125, 155], 'class_name': 'dog'},
        
        # Bird examples (K-shot)
        {'image_path': 'data/support/bird1.jpg', 'bbox': [60, 70, 80, 90], 'class_name': 'bird'},
        {'image_path': 'data/support/bird2.jpg', 'bbox': [65, 75, 85, 95], 'class_name': 'bird'},
    ]
    
    # Run detection
    detections = inferencer.detect(
        support_set=support_set,
        query_image='data/query/mixed_scene.jpg',
        score_threshold=0.4
    )
    
    print(f"Detected {len(detections)} objects in the scene")


def example_batch_detection():
    """Example: Batch processing multiple query images"""
    
    inferencer = FSODInference(
        model_path='checkpoints/best_model.pth',
        device='cuda'
    )
    
    # Support set
    support_set = [
        {'image_path': 'data/support/person1.jpg', 'bbox': [100, 120, 200, 300], 'class_name': 'person'},
        {'image_path': 'data/support/person2.jpg', 'bbox': [110, 130, 210, 310], 'class_name': 'person'},
        {'image_path': 'data/support/person3.jpg', 'bbox': [105, 125, 205, 305], 'class_name': 'person'},
    ]
    
    # Multiple query images
    query_images = [
        'data/query/image1.jpg',
        'data/query/image2.jpg',
        'data/query/image3.jpg',
    ]
    
    # Batch detection
    results = inferencer.detect_batch(
        support_set=support_set,
        query_images=query_images,
        score_threshold=0.35
    )
    
    # Save batch results
    with open('batch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} images")


def example_output_format():
    """Example of expected output format"""
    
    example_output = {
        "image_path": "data/query/test_image.jpg",
        "detections": [
            {
                "bbox": [150.5, 200.3, 250.8, 350.6],  # [x, y, width, height]
                "similarity_score": 0.8523,
                "class_name": "cat"
            },
            {
                "bbox": [400.2, 150.7, 200.5, 280.4],
                "similarity_score": 0.7891,
                "class_name": "cat"
            },
            {
                "bbox": [50.1, 300.9, 180.3, 220.7],
                "similarity_score": 0.6542,
                "class_name": "cat"
            }
        ]
    }
    
    print("Expected output format:")
    print(json.dumps(example_output, indent=2))


def example_programmatic_usage():
    """Example: Using inference programmatically"""
    
    # Initialize
    inferencer = FSODInference('checkpoints/best_model.pth')
    
    # Prepare support set dynamically
    support_set = []
    
    # Load support images with annotations
    support_images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    support_bboxes = [[10, 20, 100, 150], [15, 25, 105, 155], [12, 22, 102, 152]]
    class_name = 'target_object'
    
    for img_path, bbox in zip(support_images, support_bboxes):
        support_set.append({
            'image_path': img_path,
            'bbox': bbox,
            'class_name': class_name
        })
    
    # Run detection
    detections = inferencer.detect(support_set, 'query.jpg')
    
    # Process detections
    for det in detections:
        x, y, w, h = det['bbox']
        score = det['similarity_score']
        
        if score > 0.5:  # High confidence
            print(f"Found {class_name} at ({x:.1f}, {y:.1f}) with score {score:.3f}")


if __name__ == "__main__":
    print("=== Example 1: Single Detection ===")
    # example_single_detection()
    
    print("\n=== Example 2: Multi-class Detection ===")
    # example_multi_class_detection()
    
    print("\n=== Example 3: Batch Detection ===")
    # example_batch_detection()
    
    print("\n=== Example 4: Expected Output Format ===")
    example_output_format()
    
    print("\n=== Example 5: Programmatic Usage ===")
    # example_programmatic_usage()
    
    print("\nUncomment the examples you want to run!")