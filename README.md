# WaymoOpenMotionPred2023 - Motion Prediction
### Simple CNN

A simple solution to the Waymo motion prediction competition by rasterizing the input tabular data and using a CNN backbone and K-Means clustering to predict agents future motion.

#### Objective:
Given agents' tracks for the past 1 second on a corresponding map, predict the positions of up to 8 agents for 8 seconds into the future. 

### Solution:

### Running the code

The code are ran on notebooks to make use of Google Colab GPUs. 
To run the code: 
1. Start by download the train and validation data at:
2. Run the render.ipynb notebook to prerender raster images in blabla* format. Change the folders to your Google Drive folder with saved data.
3. Run the train.ipynb to train the model
4. *To submit to waymo competition the submit.ipynb can be

## Inspiration

The solution is inspired by many of the top solutions to Kaggle Lyft motion prediction challenge: https://www.kaggle.com/competitions/lyft-motion-prediction-autonomous-vehicles/overview
as well as earlier WaymoOpenDataset competitors such as: https://github.com/stepankonev/MotionCNN-Waymo-Open-Motion-Dataset
