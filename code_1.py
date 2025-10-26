# tomato_leaf_detection_complete.py
# Enhanced AI-IoT Smart Agriculture System
# Detect disease, estimate severity, recommend treatment, evaluate metrics, and log results

import os
from ultralytics import YOLO
import cv2
import pandas as pd
import datetime

# -------------------------------
# STEP 1: Load Trained Model
# -------------------------------
model_path = "runs/detect/tomato_leaf_detector/weights/best.pt"
model = YOLO(model_path)
print(f"✅ Loaded trained model from {model_path}")



# -------------------------------
# STEP 2: Evaluate Model Metrics
def evaluate_model(model):
    print("\n📊 Evaluating model performance...")
    
    # Pass data if needed, or use default from training
    metrics = model.val(data='data.yaml')  

    # metrics.box contains Metric object
    box_metrics = metrics.box

    # Mean metrics
    mean_precision = float(box_metrics.p.mean())
    mean_recall = float(box_metrics.r.mean())
    mean_f1 = float(box_metrics.f1.mean())
    map50 = float(box_metrics.map50)
    map50_95 = float(box_metrics.map)

    # Print results
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall:    {mean_recall:.4f}")
    print(f"Mean F1-score:  {mean_f1:.4f}")
    print(f"mAP@0.5:        {map50:.4f}")
    print(f"mAP@0.5:0.95:   {map50_95:.4f}")

    # Save evaluation images if exist
    results_dir = "runs/detect/tomato_leaf_detector"
    for img_name, desc in [("results.png","Training Curves"), ("confusion_matrix.png","Confusion Matrix")]:
        img_path = os.path.join(results_dir, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            save_path = f"eval_{img_name}"
            cv2.imwrite(save_path, img)
            print(f"Saved {desc} to {save_path}")

    print("✅ Model evaluation completed.")


# Call evaluation
evaluate_model(model)

# -------------------------------
# STEP 3: Define Disease Actions and Severity
# -------------------------------
disease_actions = {
    'Bacterial Spot': 'Apply copper-based fungicide. Avoid overhead watering.',
    'Early_Blight': 'Use chlorothalonil-based fungicide and remove infected leaves.',
    'Healthy': 'No action required. Maintain balanced nutrients.',
    'Late_blight': 'Spray mancozeb or chlorothalonil immediately.',
    'Leaf Mold': 'Improve air circulation; reduce humidity; apply fungicide if needed.',
    'Target_Spot': 'Use azoxystrobin or copper fungicide and prune lower leaves.',
    'black spot': 'Apply sulfur spray and prevent prolonged leaf wetness.'
}

def classify_severity(area_ratio):
    if area_ratio < 0.01:
        return "Mild Infection"
    elif area_ratio < 0.05:
        return "Moderate Infection"
    else:
        return "Severe Infection"

# -------------------------------
# STEP 4: Prepare Test Images & CSV
# -------------------------------
test_folder = "./test/images"       # Folder with test images
output_folder = "test_results"       # Folder to save images with boxes
os.makedirs(output_folder, exist_ok=True)

csv_file = "disease_records.csv"
# Create CSV if it doesn't exist
if not os.path.exists(csv_file):
    df_init = pd.DataFrame(columns=['timestamp','image','disease','severity','action'])
    df_init.to_csv(csv_file, index=False)

# -------------------------------
# STEP 5: Run Inference on All Test Images
# -------------------------------
test_images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]

for img_name in test_images:
    img_path = os.path.join(test_folder, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    results = model(img_path)[0]  # Run inference

    total_area = 0
    detected_classes = []

    # Draw boxes and calculate infected area
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = results.names[cls]
        detected_classes.append(label)

        bbox_area = (x2 - x1) * (y2 - y1)
        total_area += bbox_area / (w * h)

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Determine disease, severity, and action
    if detected_classes:
        predicted_disease = detected_classes[0]
        severity = classify_severity(total_area)
        action = disease_actions.get(predicted_disease, "No recommendation available")
    else:
        predicted_disease = "Healthy / Unknown"
        severity = "None"
        action = "No action needed"

    # Save output image
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, img)

    # Log result to CSV
    df = pd.DataFrame([{
        'timestamp': datetime.datetime.now(),
        'image': img_name,
        'disease': predicted_disease,
        'severity': severity,
        'action': action
    }])
    df.to_csv(csv_file, mode='a', index=False, header=False)

    print(f"✅ Processed {img_name} | Disease: {predicted_disease} | Severity: {severity}")

print(f"\n🎉 All images processed. Results saved in '{output_folder}' and logged in '{csv_file}'")
