import numpy as np
import pandas as pd
import cv2
import os



def load_params(data, train = True):
   #data = tf.io.parse_single_example(data, features_description)
    params_dict = {
        'scenario_id': data["scenario/id"].numpy(),
        'agent_id':  data["state/id"].numpy(),
        'tracks_to_pred': data["state/tracks_to_predict"].numpy(),
        'width' : data["state/current/width"].numpy(),
        'length': data["state/current/length"].numpy(),
        'agent_type': data["state/type"].numpy(),
        'past_x': data["state/past/x"].numpy(),
        'past_y': data["state/past/y"].numpy(),
        'past_yaw': data["state/past/bbox_yaw"].numpy(),
        'past_valid': data["state/past/valid"].numpy(),
        'current_x': data["state/current/x"].numpy(),
        'current_y': data["state/current/y"].numpy(),
        'current_yaw': data["state/current/bbox_yaw"].numpy(),
        'current_valid': data["state/current/valid"].numpy(),

        'road_xyz': data["roadgraph_samples/xyz"].numpy(),
        'road_type': data["roadgraph_samples/type"].numpy(),
        'road_valid': data["roadgraph_samples/valid"].numpy(),
        'road_id': data["roadgraph_samples/id"].numpy(),

        'tl_state': data["traffic_light_state/current/state"].numpy(),
        'tl_id': data["traffic_light_state/current/id"].numpy(),
        'tl_valid': data["traffic_light_state/current/valid"].numpy()
        }

    if train:

        tmp_dict = {'future_x': data["state/future/x"].numpy(),
                    'future_y': data["state/future/y"].numpy(),
                    'future_valid': data["state/future/valid"].numpy()}
        params_dict.update(tmp_dict)


    return params_dict



def r_mat(yaw):
    r_mat = np.array([
                [np.cos(-yaw), -np.sin(-yaw)],
                [np.sin(-yaw), np.cos(-yaw)]])
    return r_mat

from google.colab.patches import cv2_imshow

def rasterize(params, track_to_pred, train = 'True'):
  #Input params, output rastered params ((13,224,224), gt)

  # (1:3,224,224) represent roadgraphs and trafic lights
  # (4:8,224,224) Track to predict agent history xy
  # (9:13,224,224) All other agents history xy


  raster_size = 224
  ego_center=np.array([0.25, 0.5])
  pixel_scale=0.6
  hist_channels = 5

  agents_xy = np.concatenate(
        (np.expand_dims(np.concatenate((params['past_x'],params['current_x']), axis=1), axis=-1),
            np.expand_dims(np.concatenate((params['past_y'], params['current_y']), axis=1), axis=-1)
        ),
        axis=-1,
    )

  agents_valid = np.concatenate((params['past_valid'], params['current_valid']), axis=1)
  agents_id = params['agent_id']

  if train:
    gt_xy = np.concatenate(
          (np.expand_dims(params['future_x'], axis=-1), np.expand_dims(params['future_y'], axis=-1)), axis=-1
      )
  else:
    trans_gt_xy = []


  agents_yaw = np.concatenate((params['past_yaw'], params['current_yaw']), axis=1)


  yaw = agents_yaw[:,10][track_to_pred > 0]



  road_xy = (params['road_xyz'][:, :2][params['road_valid'].reshape(-1) > 0])
  road_types = params['road_type'][params['road_valid'] > 0]
  road_id = params['road_id'][params['road_valid'] > 0]


  rot_agents_xy = (agents_xy/pixel_scale) @ np.transpose(r_mat(yaw))
  shift_xy = rot_agents_xy[:,10][track_to_pred > 0] - raster_size*ego_center
  trans_agents_xy = rot_agents_xy - shift_xy

  shift_gt = []


  if train:
    trans_gt_xy = ((gt_xy/pixel_scale) @ np.transpose(r_mat(yaw))) - shift_xy
    #print(trans_gt_xy.shape)
    #shift_gt = raster_size*ego_center
    shift_gt = trans_gt_xy[:,0,:]
    shift_gt = np.repeat(shift_gt[:, np.newaxis,:], 80, axis=1)
    trans_gt_xy = trans_gt_xy - shift_gt
  trans_road_xy = np.squeeze(((road_xy/pixel_scale)  @ np.transpose(r_mat(yaw)) - shift_xy))
  trans_agents_yaw = agents_yaw-yaw




  road_channels = np.zeros((raster_size, raster_size, 3), dtype=np.uint8)

  track_channels = [
      np.zeros((raster_size, raster_size,1), dtype=np.uint8)
      for _ in range(hist_channels)
  ]
  agent_channels = [
      np.zeros((raster_size, raster_size,1), dtype=np.uint8)
      for _ in range(hist_channels)
  ]


  RoadColorMap = {1:16, 2:16,3:16,
                11: 28,
                7 : 40,
                9: 64,
                6: 88,
                12: 112,
                8: 136,
                10: 160,
                13: 184,
                16: 214,
                15: 255,
                17: 509,
                18: 459,
                19: 479,
                4:128,
                5:128,
                14:128,
                20:128
              }


  for id in np.unique(road_id):
    roadline = trans_road_xy[road_id == id]
    line_type = road_types[road_id == id]
    color = pd.Series(line_type).map(RoadColorMap)
    color[np.isnan(color)] = 0


    if color[0] < 256:
      road_channels = cv2.polylines(
                      road_channels,
                      [roadline.astype(int)],
                      False,
                  (0,0,int(color[0]))

                  )
    else:
       road_channels = cv2.polylines(
                      road_channels,
                      [roadline.astype(int)],
                      False,
                  (int(color[0])-255,0,0)
                  )

  TlColorMap = {0:0,
                1:255,
                2:383,
                3: 510,
                4: 255,
                5: 383,
                6: 510,
                7: 255,
                8: 383,
              }

  for id in np.unique(params['tl_id']):
      roadline = trans_road_xy[road_id == id]
      line_type = params['tl_state'][params['tl_id'] == id]
      color = pd.Series(line_type).map(TlColorMap)
      color[np.isnan(color)] = 0


      if color[0] < 256:
        road_channels = cv2.polylines(
                        road_channels,
                        [roadline.astype(int)],
                        False,
                    (int(color[0]),0,0)

                    )
      else:
        road_channels = cv2.polylines(
                        road_channels,
                        [roadline.astype(int)],
                        False,
                    (0,int(color[0])-255,0)
                    )


  
  ego_type = params['agent_type'][track_to_pred > 0]
  col_adj = int(ego_type) * 10
  ego_id = agents_id[track_to_pred>0]
  ego_xy = trans_agents_xy[:,:][track_to_pred > 0]
  ego_w = (np.squeeze(params['width'][track_to_pred > 0]))/pixel_scale
  ego_l = (np.squeeze(params['length'][track_to_pred > 0]))/pixel_scale
  bbpoints = np.array([[-ego_l/2,  ego_w/2], [ego_l/2 , ego_w/2], [ego_l/2, -ego_w/2], [-ego_l/2, -ego_w/2]], np.float32)
  for i in range(hist_channels):
    for j in range(3):
      angle = trans_agents_yaw[:,10-(j+(i*2))][track_to_pred > 0]

      t_bbpoints = (bbpoints @ np.transpose(r_mat(-angle))) + ego_xy[0,10-(j+(i*2))]
      cv2.fillPoly(
                        track_channels[i],
                        t_bbpoints.astype(int),
                        color= 255-col_adj-(65*j))
                         #color= 60+(45*(i))-(15*(2-j))
                         



  agents_w = (np.squeeze(params['width']))/pixel_scale
  agents_l = (np.squeeze(params['length']))/pixel_scale
  unique_agents_id = np.unique(agents_id[agents_valid[:,10] > 0])

  for id in (unique_agents_id):
    if id != ego_id:
      bbpoints = np.squeeze(np.array([[-agents_l[agents_id == id]/2, agents_w[agents_id == id]/2],
                          [agents_l[agents_id == id]/2 , agents_w[agents_id == id]/2],
                          [agents_l[agents_id == id]/2, -agents_w[agents_id == id]/2],
                          [-agents_l[agents_id == id]/2, -agents_w[agents_id == id]/2]], np.float32))

      for i in range(hist_channels):
        for j in range(3):
          angle = trans_agents_yaw[:,10-(j+(i*2))][agents_id == id][0]

          t_bbpoints = (bbpoints @ np.transpose(r_mat(-angle))) + trans_agents_xy[:,10-(j+(i*2))][agents_id == id]

          if ((t_bbpoints.max() < raster_size) & (t_bbpoints.min() > 0)):
            cv2.polylines(
                              agent_channels[i],
                              [np.int32([t_bbpoints])],
                              True,
                              color= 255-(65*(j)))




    raster = np.concatenate([road_channels] + track_channels + agent_channels, axis=2)

    raster_dict = {
            "track_to_pred": track_to_pred,
            "raster": raster,
            "yaw": yaw,
            "shift": shift_xy,
            "gt": trans_gt_xy[track_to_pred > 0],
            "gt_shift": shift_gt,
            "scenario_id": params['scenario_id'],
            "ego_type": ego_type,
            "valid": np.squeeze(params['future_valid'][track_to_pred > 0])
    }


  return raster_dict



def save_raster(raster_dict, step):

  save_dir = '/content/traintar/'

  filename = f"{raster_dict['ego_type']}_{raster_dict['scenario_id']}_{step}.npz"
  np.savez_compressed(os.path.join(save_dir, filename), **raster_dict)