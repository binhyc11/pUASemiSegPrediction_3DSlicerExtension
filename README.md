# pUASemiSegPrediction_3DSlicerExtension
A complete workflow for predicting pure uric acid urinary stones that includes semi-auto segmentation stones, radiomics feature extraction, and machine learning prediction.

- v2.2: April 15th, 2025:
	+ Separate the Semi-auto segmentation task and the Prediction task.
	+ Allow users to adjust the semi-auto segmented ROI before getting the prediction.

- v2.1: April 9th, 2025:
	+ Use pickle files of scaler and trained model instead of raw training data in json.

- v2.0: Dec 20th, 2024:
	+ First working 3D Slicer extension.
