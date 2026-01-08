# How to run:

python -u main.py --geo_graph --transtime --temp_encoder gru --agew


# synthetic_data:
- The time dimension is T=8 years (2012-2019), the number of stations is N=306. 
- Note: N=306 is the global station count up to 2024, not all stations opened before 2019.

## Station IDs
- 201812.csv: The ID of stations that opened before 201812.
- 201912.csv: The ID of stations that opened before 201912.

## OD flows
- year_od_workday_syn_19.npy: (T, N, N)
  Yearly OD flows for workdays (synthetic).
- year_od_weekend_syn_19.npy: (T, N, N)
  Yearly OD flows for weekends (synthetic).

## Station features
- year_sta_features.npy: (T, N, 29)
  Station-level features. 

## OD interaction features
- od_interfea.npy: (T, N, N, 4)
  OD-level interaction features (4 channels).

## Masks
- exist_sta_mask.npy: (T, N)
  Station existence mask per year (1: exists, 0: not exists).
- exist_od_mask.npy: (T, N, N)
  OD existence mask (1: valid OD pair, 0: invalid).


## KNN indices for localized graphs
- geo_dist_knn_metro.npy: (T, N, 6)
- link_dist_knn_metro.npy: (T, N, 6)
- func_knn_metro.npy: (T, N, 6)
- trans_time_knn_metro.npy: (T, N, 6)

Each file stores k-nearest neighbors (k=5) plus self (total 6 indices) for each station,
built with different distance/graph definitions.
