# Evaluation of TTM in the GIFT-Eval Leaderboard

The [GIFT-Eval Leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval) is a comprehensive benchmark for time series forecasting.
Here, we provide the details of evaluating [Tiny Time Mixers (TTMs)](https://arxiv.org/abs/2401.03955) on the GIFT benchmark.

TTMs achive the current state-of-art performance in point forecasting (normalized MASE of 0.679), with average finetune time of only 2.5 minutes in one A100 GPU. Details of the evaluation framework is provided below.

## Few-shot finetune
TTMs are lightweight and extremely fast, making them suitable for fine-tuning on the target domain data.
In the GIFT-Eval benchmark, we finetune TTMs independently on each dataset, with only 20% of the training data for most of the datasets (for extremely short datasets with <200 finetune samples, we use 90% few-shot setting).

Each dataset is chronologically split into train, validation, and test datasets. TTMs are finetuned on 20% random windows taken from the training split, validated on the validation split, and then finally the performance on the test split has been reported.

## TTM Version
TTM r2 models have been used in this evaluation. See the [model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) here.

## Results
| dataset                          |   MASE |   sMAPE |   NRMSE |    ND |   CRPS |   Finetune time (s) |
|:---------------------------------|-------:|--------:|--------:|------:|-------:|--------------------:|
| bitbrains_fast_storage/5T/long   |  0.939 |   0.839 |   5.624 | 1.007 |  0.937 |            2202.97  |
| bitbrains_fast_storage/5T/medium |  1.069 |   0.83  |   5.477 | 0.888 |  0.827 |            2672.43  |
| bitbrains_fast_storage/5T/short  |  0.736 |   0.744 |   4.644 | 0.565 |  0.563 |            3059.45  |
| bitbrains_fast_storage/H/short   |  1.203 |   0.628 |   5.941 | 1.153 |  1.03  |             139.66  |
| bitbrains_rnd/5T/long            |  3.503 |   0.753 |   6.082 | 0.799 |  0.762 |             889.592 |
| bitbrains_rnd/5T/medium          |  4.53  |   0.756 |   6.76  | 0.796 |  0.783 |            1056.78  |
| bitbrains_rnd/5T/short           |  1.753 |   0.67  |   4.852 | 0.465 |  0.467 |            1185.68  |
| bitbrains_rnd/H/short            |  6.063 |   0.677 |   6.089 | 0.963 |  0.892 |              56.082 |
| bizitobs_application/10S/long    |  3.805 |   0.056 |   0.103 | 0.062 |  0.058 |             127.735 |
| bizitobs_application/10S/medium  |  3.053 |   0.045 |   0.093 | 0.053 |  0.049 |             131.327 |
| bizitobs_application/10S/short   |  1.646 |   0.025 |   0.046 | 0.026 |  0.022 |              38.841 |
| bizitobs_l2c/5T/long             |  0.505 |   0.7   |   0.444 | 0.286 |  0.242 |             148.742 |
| bizitobs_l2c/5T/medium           |  0.548 |   0.633 |   0.471 | 0.292 |  0.247 |             151.259 |
| bizitobs_l2c/5T/short            |  0.247 |   0.195 |   0.138 | 0.082 |  0.069 |             164.778 |
| bizitobs_l2c/H/long              |  1.204 |   0.908 |   0.906 | 0.716 |  0.693 |               0     |
| bizitobs_l2c/H/medium            |  1.138 |   0.916 |   0.891 | 0.698 |  0.675 |               0     |
| bizitobs_l2c/H/short             |  0.544 |   0.646 |   0.477 | 0.299 |  0.269 |             131.714 |
| bizitobs_service/10S/long        |  1.484 |   0.097 |   0.272 | 0.061 |  0.059 |             303.533 |
| bizitobs_service/10S/medium      |  1.258 |   0.085 |   0.207 | 0.045 |  0.043 |             295.509 |
| bizitobs_service/10S/short       |  0.8   |   0.067 |   0.05  | 0.016 |  0.014 |             310.131 |
| car_parts/M/short                |  0.839 |   1.97  |   2.882 | 1.06  |  1.103 |              65.994 |
| covid_deaths/D/short             | 30.802 |   0.354 |   0.25  | 0.04  |  0.036 |              82.347 |
| electricity/15T/long             |  0.869 |   0.134 |   0.915 | 0.092 |  0.078 |            8915.89  |
| electricity/15T/medium           |  0.822 |   0.132 |   0.874 | 0.091 |  0.077 |            9275.99  |
| electricity/15T/short            |  0.909 |   0.158 |   0.682 | 0.096 |  0.082 |            9649.3   |
| electricity/D/short              |  1.45  |   0.101 |   0.6   | 0.072 |  0.058 |             377.818 |
| electricity/H/long               |  1.191 |   0.145 |   1.084 | 0.106 |  0.09  |            2031.31  |
| electricity/H/medium             |  1.088 |   0.135 |   0.897 | 0.093 |  0.078 |            2014.42  |
| electricity/H/short              |  0.901 |   0.126 |   0.588 | 0.079 |  0.068 |            2266.28  |
| electricity/W/short              |  1.523 |   0.101 |   0.66  | 0.073 |  0.06  |             174.701 |
| ett1/15T/long                    |  1.036 |   0.405 |   0.553 | 0.309 |  0.255 |             374.458 |
| ett1/15T/medium                  |  1.071 |   0.411 |   0.604 | 0.324 |  0.272 |             377.527 |
| ett1/15T/short                   |  0.702 |   0.238 |   0.423 | 0.209 |  0.175 |             400.692 |
| ett1/D/short                     |  1.878 |   0.508 |   0.612 | 0.411 |  0.342 |             177.608 |
| ett1/H/long                      |  1.392 |   0.481 |   0.565 | 0.346 |  0.281 |              77.482 |
| ett1/H/medium                    |  1.284 |   0.45  |   0.559 | 0.337 |  0.273 |              80.172 |
| ett1/H/short                     |  0.849 |   0.267 |   0.457 | 0.237 |  0.196 |              88.371 |
| ett1/W/short                     |  1.595 |   0.541 |   0.548 | 0.382 |  0.283 |               0     |
| ett2/15T/long                    |  0.929 |   0.172 |   0.184 | 0.118 |  0.095 |             370.414 |
| ett2/15T/medium                  |  0.916 |   0.168 |   0.192 | 0.117 |  0.096 |             357.348 |
| ett2/15T/short                   |  0.727 |   0.131 |   0.127 | 0.078 |  0.064 |             383.802 |
| ett2/D/short                     |  1.206 |   0.13  |   0.162 | 0.1   |  0.085 |              52.935 |
| ett2/H/long                      |  1.037 |   0.185 |   0.203 | 0.128 |  0.105 |              81.233 |
| ett2/H/medium                    |  1.03  |   0.169 |   0.205 | 0.129 |  0.104 |              86.377 |
| ett2/H/short                     |  0.749 |   0.112 |   0.132 | 0.082 |  0.066 |              96.161 |
| ett2/W/short                     |  0.785 |   0.133 |   0.157 | 0.102 |  0.094 |               0     |
| hierarchical_sales/D/short       |  0.756 |   1.075 |   1.64  | 0.718 |  0.606 |             296.193 |
| hierarchical_sales/W/short       |  0.727 |   0.463 |   0.988 | 0.413 |  0.362 |             118.517 |
| hospital/M/short                 |  0.75  |   0.171 |   0.19  | 0.064 |  0.053 |              49.097 |
| jena_weather/10T/long            |  0.768 |   0.677 |   0.224 | 0.064 |  0.068 |            1186.83  |
| jena_weather/10T/medium          |  0.717 |   0.695 |   0.236 | 0.067 |  0.069 |            1208.96  |
| jena_weather/10T/short           |  0.32  |   0.549 |   0.176 | 0.038 |  0.044 |            1342.39  |
| jena_weather/D/short             |  1.322 |   0.499 |   0.194 | 0.088 |  0.072 |              70.93  |
| jena_weather/H/long              |  1.462 |   0.751 |   0.7   | 0.211 |  0.188 |             135.956 |
| jena_weather/H/medium            |  0.861 |   0.681 |   0.205 | 0.067 |  0.07  |             103.314 |
| jena_weather/H/short             |  0.58  |   0.618 |   0.201 | 0.058 |  0.062 |             116.307 |
| kdd_cup_2018/D/short             |  1.197 |   0.466 |   1.196 | 0.47  |  0.4   |             515.278 |
| kdd_cup_2018/H/long              |  1.004 |   0.598 |   1.484 | 0.557 |  0.475 |             487.508 |
| kdd_cup_2018/H/medium            |  1.03  |   0.549 |   1.49  | 0.523 |  0.453 |             555.967 |
| kdd_cup_2018/H/short             |  0.95  |   0.503 |   1.363 | 0.473 |  0.415 |             604.517 |
| loop_seattle/5T/long             |  0.871 |   0.125 |   0.17  | 0.098 |  0.084 |            7245.76  |
| loop_seattle/5T/medium           |  0.808 |   0.117 |   0.157 | 0.091 |  0.077 |            7438.39  |
| loop_seattle/5T/short            |  0.558 |   0.075 |   0.112 | 0.06  |  0.051 |            8082.58  |
| loop_seattle/D/short             |  0.899 |   0.055 |   0.079 | 0.053 |  0.045 |             386.424 |
| loop_seattle/H/long              |  0.904 |   0.101 |   0.138 | 0.077 |  0.067 |             398.073 |
| loop_seattle/H/medium            |  0.923 |   0.103 |   0.143 | 0.08  |  0.07  |             471.572 |
| loop_seattle/H/short             |  0.877 |   0.1   |   0.138 | 0.077 |  0.067 |             524.563 |
| m_dense/D/short                  |  0.731 |   0.106 |   0.168 | 0.085 |  0.071 |             142.949 |
| m_dense/H/long                   |  0.734 |   0.212 |   0.339 | 0.153 |  0.128 |             631.487 |
| m_dense/H/medium                 |  0.709 |   0.206 |   0.333 | 0.144 |  0.122 |             659.97  |
| m_dense/H/short                  |  0.809 |   0.226 |   0.377 | 0.165 |  0.141 |             732.951 |
| m4_daily/D/short                 |  3.302 |   0.031 |   0.109 | 0.028 |  0.024 |            2935.28  |
| m4_hourly/H/short                |  1.029 |   0.124 |   0.203 | 0.041 |  0.035 |             264.646 |
| m4_monthly/M/short               |  0.946 |   0.132 |   0.28  | 0.116 |  0.1   |            2729.5   |
| m4_quarterly/Q/short             |  1.17  |   0.101 |   0.222 | 0.093 |  0.079 |             350.686 |
| m4_weekly/W/short                |  1.945 |   0.07  |   0.1   | 0.052 |  0.044 |             992.943 |
| m4_yearly/A/short                |  3.248 |   0.144 |   0.295 | 0.139 |  0.118 |             102.097 |
| restaurant/D/short               |  0.699 |   0.401 |   0.558 | 0.337 |  0.269 |             804.052 |
| saugeen/D/short                  |  2.995 |   0.343 |   1.149 | 0.437 |  0.406 |              90.953 |
| saugeen/M/short                  |  0.754 |   0.362 |   0.622 | 0.387 |  0.34  |              46.979 |
| saugeen/W/short                  |  1.373 |   0.447 |   1.065 | 0.511 |  0.445 |             150.33  |
| solar/10T/long                   |  0.997 |   1.485 |   1.061 | 0.492 |  0.486 |            1541.07  |
| solar/10T/medium                 |  0.996 |   1.489 |   1.057 | 0.502 |  0.497 |            1559.54  |
| solar/10T/short                  |  0.824 |   1.514 |   1.241 | 0.549 |  0.544 |            1648.48  |
| solar/D/short                    |  0.983 |   0.432 |   0.504 | 0.364 |  0.303 |             183.113 |
| solar/H/long                     |  1.404 |   1.459 |   1.313 | 0.637 |  0.607 |             188.01  |
| solar/H/medium                   |  1.237 |   1.447 |   1.279 | 0.582 |  0.559 |             218.911 |
| solar/H/short                    |  0.881 |   1.414 |   0.961 | 0.422 |  0.409 |             244.32  |
| solar/W/short                    |  0.996 |   0.18  |   0.249 | 0.186 |  0.171 |              88.441 |
| sz_taxi/15T/long                 |  0.588 |   0.534 |   0.402 | 0.284 |  0.239 |              19.353 |
| sz_taxi/15T/medium               |  0.545 |   0.411 |   0.377 | 0.261 |  0.215 |              13.244 |
| sz_taxi/15T/short                |  0.561 |   0.407 |   0.386 | 0.262 |  0.21  |             337.854 |
| sz_taxi/H/short                  |  0.576 |   0.308 |   0.253 | 0.176 |  0.142 |              46.291 |
| temperature_rain/D/short         |  1.413 |   1.515 |   1.642 | 0.716 |  0.656 |            7180.64  |
| us_births/D/short                |  0.388 |   0.025 |   0.041 | 0.025 |  0.02  |              41.56  |
| us_births/M/short                |  0.511 |   0.014 |   0.019 | 0.014 |  0.011 |              64.508 |
| us_births/W/short                |  1.139 |   0.017 |   0.023 | 0.017 |  0.014 |             211.301 |
| **Geometric Mean**               |**1.057**|**0.268**|**0.43**|**0.167**|**0.147**|       **148.521**|


In the above table, a finetune time of 0s indicate that zero-shot evaluation was used instead of few-shot evaluation.
Note that the scores in the GIFT-Eval leaderboard are normalized with respect to seasonal naive scores, and the scores reported in the above table are the raw scores on each dataset.