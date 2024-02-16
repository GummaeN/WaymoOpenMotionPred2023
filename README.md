# WaymoOpenMotionPred2023 - Motion Prediction
### Simple CNN
Två bilder
A simple solution to the Waymo motion prediction competition by rasterizing the input tabular data and using a CNN backbone and K-Means clustering to predict agents future motion.

#### Objective:
Given agents' tracks for the past 1 second on a corresponding map, predict the positions of up to 8 agents for 8 seconds into the future. 

### Solution:
The tf-record files including tabular data of the roadnet, agents(vehicles,bikes,pedestrians) as well as traffic lights were rasterized into images of size 13x224x224. 
First 3 channels were for roadnet and traffic lights.
The agents data were in 10Hz thus giving us 10 frames of history data as well as 1 current frame.
5 channels were used for the agent of whom we should predict, using every second frame. For every frame the two closest history frames were plotted as well to show the velocity of the agent.
The last 5 channels were used in the same way but for all other agents.

For the training data agents with 1 second of history data and at least 7 seconds of future data were chosen.

Given how time consuming the rasterization is these steps had to be done in advance and were all prerendered into npz files.

The final solution consisted of an ensamble of 4 different simple models.
Each model containing of a CNN backbone, with two final dense layers to output the confidences as well as the trajectories.

The 4 models used were:

To ensemble the models and pick 8 final trajectories for the competition all trajectories were clustered into 8 clusters using kmeans clustering. The trajecotry with highest confidence was chosen for each cluster and the confidences was normalized.

### Running the code

The code are ran on notebooks to make use of Google Colab GPUs. 
To run the code: 
1. Start by download the train and validation data at: https://waymo.com/open/download/
2. Run the render.ipynb notebook to prerender raster images in npz format. Change the folders to your Google Drive folder with saved data.
3. Run the train.ipynb to train the model
4. (work in progress)*To submit to waymo competition the test.ipynb can be ran.

## Inspiration

The solution is inspired by many of the top solutions to Kaggle Lyft motion prediction challenge: https://www.kaggle.com/competitions/lyft-motion-prediction-autonomous-vehicles/overview
as well as earlier WaymoOpenDataset competitors such as: https://github.com/stepankonev/MotionCNN-Waymo-Open-Motion-Dataset
