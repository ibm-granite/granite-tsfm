# Evaluation of TTM in the GIFT-Eval Leaderboard

The [**GIFT-Eval Leaderboard**](https://huggingface.co/spaces/Salesforce/GIFT-Eval) is a comprehensive benchmark for time series forecasting.  

[**Tiny Time Mixers (TTMs)**](https://arxiv.org/abs/2401.03955) or **TTMs** are lightweight compact pre-trained models (ranging from 1-5 Million parameters). Here, we present the evaluation of TTMs on the GIFT benchmark. TTMs achieve **state-of-the-art performance** in point forecasting, with a **normalized MASE of 0.679**, while maintaining an **average fine-tuning time of just 2.5 minutes** on a single **A100 GPU**.  

Details of the evaluation framework are provided below.

## Methodology
TTMs are **lightweight** and **extremely fast**, making them ideal for fine-tuning on target domain data.  

In the **GIFT-Eval** benchmark, we fine-tune TTMs separately for each dataset, using **only 20%** of the training data for most datasets. However, for extremely short datasets (fewer than 200 fine-tuning samples), we adopt a 90% few-shot setting. Since GIFT follows GlounsTS framework which allows the entire training data to be used as in-context, this approach can also be referred to as **in-context 20% learning**.

Each dataset is chronologically split into **train, validation, and test sets**. TTMs are fine-tuned on **random windows covering 20% of the training split**, validated on the validation set, and finally evaluated on the test set, with the reported performance reflecting this final evaluation.

## TTM Version
TTM r2 models have been used in this evaluation. See the [model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) here.

## Results
We show the normalized forecasting metrics below (normalized by seasonal naive scores as done in the GIFT-Eval leaderboard).

| dataset                          |   MASE |   sMAPE |   NRMSE |    ND |   CRPS |   Finetune time (s) |
|:---------------------------------|-------:|--------:|--------:|------:|-------:|--------------------:|
| bitbrains_fast_storage/5T/long   |  0.824 |   1.668 |   0.837 | 0.941 |  0.726 |            2202.97  |
| bitbrains_fast_storage/5T/medium |  0.876 |   1.837 |   0.791 | 0.879 |  0.651 |            2672.43  |
| bitbrains_fast_storage/5T/short  |  0.646 |   1.562 |   0.636 | 0.518 |  0.466 |            3059.45  |
| bitbrains_fast_storage/H/short   |  0.926 |   2.006 |   1.017 | 1.25  |  0.953 |             139.66  |
| bitbrains_rnd/5T/long            |  1.001 |   1.868 |   0.809 | 0.878 |  0.591 |             889.592 |
| bitbrains_rnd/5T/medium          |  0.998 |   1.928 |   0.888 | 0.923 |  0.621 |            1056.78  |
| bitbrains_rnd/5T/short           |  0.89  |   1.745 |   0.643 | 0.539 |  0.425 |            1185.68  |
| bitbrains_rnd/H/short            |  1.004 |   1.875 |   0.722 | 0.868 |  0.686 |              56.082 |
| bizitobs_application/10S/long    |  0     |   0.054 |   0.076 | 0.062 |  0.059 |             127.735 |
| bizitobs_application/10S/medium  |  1.135 |   1.121 |   1.043 | 1.049 |  1.16  |             131.327 |
| bizitobs_application/10S/short   |  0.735 |   0.712 |   0.6   | 0.627 |  0.639 |              38.841 |
| bizitobs_l2c/5T/long             |  0.348 |   0.539 |   0.386 | 0.332 |  0.358 |             148.742 |
| bizitobs_l2c/5T/medium           |  0.442 |   0.541 |   0.522 | 0.431 |  0.466 |             151.259 |
| bizitobs_l2c/5T/short            |  0.25  |   0.385 |   0.305 | 0.247 |  0.263 |             164.778 |
| bizitobs_l2c/H/long              |  0.298 |   0.818 |   0.492 | 0.434 |  0.381 |               0     |
| bizitobs_l2c/H/medium            |  0.69  |   0.619 |   0.696 | 0.738 |  0.475 |               0     |
| bizitobs_l2c/H/short             |  0.449 |   0.566 |   0.527 | 0.443 |  0.501 |             131.714 |
| bizitobs_service/10S/long        |  1.084 |   1.194 |   1.015 | 1.099 |  1.047 |             303.533 |
| bizitobs_service/10S/medium      |  0.953 |   1.101 |   0.885 | 0.935 |  0.876 |             295.509 |
| bizitobs_service/10S/short       |  0.651 |   0.882 |   0.25  | 0.402 |  0.353 |             310.131 |
| car_parts/M/short                |  0.699 |   1.173 |   0.76  | 0.662 |  0.641 |              65.994 |
| covid_deaths/D/short             |  0.657 |   1.99  |   0.315 | 0.302 |  0.291 |              82.347 |
| electricity/15T/long             |  0.749 |   0.852 |   0.88  | 0.794 |  0.602 |            8915.89  |
| electricity/15T/medium           |  0.715 |   0.813 |   0.832 | 0.745 |  0.621 |            9275.99  |
| electricity/15T/short            |  0.529 |   0.687 |   0.573 | 0.515 |  0.497 |            9649.3   |
| electricity/D/short              |  0.729 |   0.746 |   0.571 | 0.629 |  0.477 |             377.818 |
| electricity/H/long               |  0.783 |   0.865 |   0.747 | 0.754 |  0.474 |            2031.31  |
| electricity/H/medium             |  0.783 |   0.86  |   0.793 | 0.776 |  0.502 |            2014.42  |
| electricity/H/short              |  0.662 |   0.763 |   0.733 | 0.679 |  0.624 |            2266.28  |
| electricity/W/short              |  0.729 |   0.842 |   0.512 | 0.563 |  0.607 |             174.701 |
| ett1/15T/long                    |  0.871 |   0.964 |   0.803 | 0.872 |  0.644 |             374.458 |
| ett1/15T/medium                  |  0.9   |   0.959 |   0.884 | 0.925 |  0.774 |             377.527 |
| ett1/15T/short                   |  0.751 |   0.778 |   0.755 | 0.738 |  0.725 |             400.692 |
| ett1/D/short                     |  1.055 |   0.985 |   0.968 | 1.011 |  0.665 |             177.608 |
| ett1/H/long                      |  0.94  |   1.008 |   0.836 | 0.934 |  0.457 |              77.482 |
| ett1/H/medium                    |  0.818 |   0.985 |   0.816 | 0.862 |  0.506 |              80.172 |
| ett1/H/short                     |  0.869 |   0.931 |   0.823 | 0.872 |  0.782 |              88.371 |
| ett1/W/short                     |  0.901 |   0.866 |   1.099 | 0.989 |  0.836 |               0     |
| ett2/15T/long                    |  0.92  |   0.912 |   0.936 | 0.934 |  0.578 |             370.414 |
| ett2/15T/medium                  |  0.873 |   0.882 |   0.93  | 0.887 |  0.673 |             357.348 |
| ett2/15T/short                   |  0.679 |   0.768 |   0.699 | 0.649 |  0.662 |             383.802 |
| ett2/D/short                     |  0.868 |   0.913 |   0.808 | 0.825 |  0.414 |              52.935 |
| ett2/H/long                      |  0.917 |   0.983 |   0.922 | 0.932 |  0.365 |              81.233 |
| ett2/H/medium                    |  0.83  |   0.855 |   0.825 | 0.835 |  0.433 |              86.377 |
| ett2/H/short                     |  0.812 |   0.82  |   0.848 | 0.816 |  0.702 |              96.161 |
| ett2/W/short                     |  1.008 |   1.044 |   0.864 | 0.917 |  0.556 |               0     |
| hierarchical_sales/D/short       |  0.669 |   0.903 |   0.695 | 0.678 |  0.257 |             296.193 |
| hierarchical_sales/W/short       |  0.705 |   0.798 |   0.668 | 0.663 |  0.352 |             118.517 |
| hospital/M/short                 |  0.815 |   0.816 |   0.89  | 0.883 |  0.841 |              49.097 |
| jena_weather/10T/long            |  1.009 |   1.675 |   0.81  | 0.919 |  0.223 |            1186.83  |
| jena_weather/10T/medium          |  1.001 |   1.728 |   0.757 | 0.874 |  0.248 |            1208.96  |
| jena_weather/10T/short           |  0.431 |   1.392 |   0.635 | 0.583 |  0.285 |            1342.39  |
| jena_weather/D/short             |  0.842 |   1.029 |   1.275 | 1.11  |  0.243 |              70.93  |
| jena_weather/H/long              |  1.151 |   1.496 |   2.229 | 2.231 |  0.314 |             135.956 |
| jena_weather/H/medium            |  0.968 |   1.496 |   0.758 | 0.863 |  0.144 |             103.314 |
| jena_weather/H/short             |  0.802 |   1.54  |   0.745 | 0.855 |  0.359 |             116.307 |
| kdd_cup_2018/D/short             |  0.798 |   0.815 |   0.949 | 0.878 |  0.45  |             515.278 |
| kdd_cup_2018/H/long              |  0.749 |   0.739 |   0.905 | 0.801 |  0.38  |             487.508 |
| kdd_cup_2018/H/medium            |  0.72  |   0.839 |   0.937 | 0.797 |  0.478 |             555.967 |
| kdd_cup_2018/H/short             |  0.709 |   0.756 |   0.868 | 0.701 |  0.742 |             604.517 |
| loop_seattle/5T/long             |  0.697 |   0.712 |   0.731 | 0.696 |  0.611 |            7245.76  |
| loop_seattle/5T/medium           |  0.703 |   0.723 |   0.72  | 0.703 |  0.63  |            7438.39  |
| loop_seattle/5T/short            |  0.732 |   0.734 |   0.744 | 0.732 |  0.634 |            8082.58  |
| loop_seattle/D/short             |  0.519 |   0.512 |   0.577 | 0.501 |  0.347 |             386.424 |
| loop_seattle/H/long              |  0.583 |   0.617 |   0.599 | 0.577 |  0.275 |             398.073 |
| loop_seattle/H/medium            |  0.624 |   0.642 |   0.653 | 0.623 |  0.338 |             471.572 |
| loop_seattle/H/short             |  0.679 |   0.707 |   0.699 | 0.685 |  0.62  |             524.563 |
| m_dense/D/short                  |  0.438 |   0.412 |   0.487 | 0.413 |  0.24  |             142.949 |
| m_dense/H/long                   |  0.496 |   0.545 |   0.528 | 0.503 |  0.233 |             631.487 |
| m_dense/H/medium                 |  0.452 |   0.519 |   0.482 | 0.437 |  0.254 |             659.97  |
| m_dense/H/short                  |  0.543 |   0.578 |   0.557 | 0.514 |  0.503 |             732.951 |
| m4_daily/D/short                 |  1.007 |   1.009 |   1.003 | 1.017 |  0.891 |            2935.28  |
| m4_hourly/H/short                |  0.864 |   0.89  |   0.779 | 0.847 |  0.874 |             264.646 |
| m4_monthly/M/short               |  0.751 |   0.825 |   0.825 | 0.797 |  0.795 |            2729.5   |
| m4_quarterly/Q/short             |  0.731 |   0.807 |   0.843 | 0.785 |  0.799 |             350.686 |
| m4_weekly/W/short                |  0.7   |   0.765 |   0.816 | 0.822 |  0.604 |             992.943 |
| m4_yearly/A/short                |  0.818 |   0.88  |   0.91  | 0.86  |  0.858 |             102.097 |
| restaurant/D/short               |  0.692 |   0.721 |   0.738 | 0.688 |  0.296 |             804.052 |
| saugeen/D/short                  |  0.878 |   0.809 |   1.126 | 0.877 |  0.539 |              90.953 |
| saugeen/M/short                  |  0.773 |   0.82  |   0.776 | 0.773 |  0.765 |              46.979 |
| saugeen/W/short                  |  0.69  |   0.802 |   0.832 | 0.689 |  0.52  |             150.33  |
| solar/10T/long                   |  1.144 |   2.25  |   0.947 | 1.141 |  0.619 |            1541.07  |
| solar/10T/medium                 |  1.075 |   2.259 |   0.881 | 1.068 |  0.645 |            1559.54  |
| solar/10T/short                  |  0.743 |   1.637 |   0.693 | 0.747 |  0.632 |            1648.48  |
| solar/D/short                    |  0.848 |   0.867 |   0.817 | 0.854 |  0.4   |             183.113 |
| solar/H/long                     |  1.312 |   2.049 |   1.113 | 1.314 |  0.413 |             188.01  |
| solar/H/medium                   |  1.323 |   2.547 |   1.122 | 1.322 |  0.44  |             218.911 |
| solar/H/short                    |  0.926 |   2.281 |   0.821 | 0.932 |  0.651 |             244.32  |
| solar/W/short                    |  0.678 |   0.711 |   0.727 | 0.679 |  0.726 |              88.441 |
| sz_taxi/15T/long                 |  0.851 |   1.023 |   0.814 | 0.833 |  0.431 |              19.353 |
| sz_taxi/15T/medium               |  0.765 |   0.796 |   0.752 | 0.762 |  0.474 |              13.244 |
| sz_taxi/15T/short                |  0.734 |   0.756 |   0.738 | 0.734 |  0.681 |             337.854 |
| sz_taxi/H/short                  |  0.78  |   0.8   |   0.783 | 0.778 |  0.618 |              46.291 |
| temperature_rain/D/short         |  0.703 |   1.139 |   0.793 | 0.77  |  0.403 |            7180.64  |
| us_births/D/short                |  0.209 |   0.204 |   0.26  | 0.208 |  0.142 |              41.56  |
| us_births/M/short                |  0.672 |   0.675 |   0.719 | 0.675 |  0.681 |              64.508 |
| us_births/W/short                |  0.73  |   0.731 |   0.723 | 0.727 |  0.66  |             211.301 |
| **Geometric Mean**                   |  **0.679** |   **0.91**  |   **0.728** | **0.719** |  **0.492** |             **148.521** |

In the table above, a fine-tune time of 0 seconds indicates that zero-shot evaluation was performed instead of few-shot evaluation. This is because the number of time series segments available for fine-tuning the TTM model was very low (fewer than 10).