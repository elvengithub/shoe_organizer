Training classes for the camera "type" head live here: one subfolder per label.

1. Create a folder per class (names can be your own, e.g. nike_trail, chelsea_boot).
2. List those names in class_order.txt (same order as when you train).
3. Add each name under ai_pipeline.type_to_bucket in config.yaml as sports or casual,
   or rely on type_to_bucket_default for new names.
4. Retrain: python scripts/train_torch_classifier.py --data dataset/type --out models/shoe_type_classifier.pt

The live camera shows the predicted folder name as shoe_dataset_style / in shoe_type_label.
