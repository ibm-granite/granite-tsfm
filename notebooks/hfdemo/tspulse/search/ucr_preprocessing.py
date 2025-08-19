import os
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler

from tsfm_public.toolkit.dataset import ClassificationDFDataset
from tsfm_public.toolkit.time_series_classification_preprocessor import (
    TimeSeriesClassificationPreprocessor,
)
from tsfm_public.toolkit.util import convert_tsfile_to_dataframe


np.random.seed(0)


DATA_DIR = "./Univariate_ts"
DEFAULT_LIST_UCR_DSETS = [
    "ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "BME",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "CBF",
    "Car",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "Earthquakes",
    "ElectricDevices",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OSULeaf",
    "OliveOil",
    "PLAID",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]


LABEL_COLUMN = "class_vals"
INPUT_COLUMNS = ["dim_0"]  # because univariate ts
CONTEXT_LENGTH = 512


min_num_req_samples = 10
mixed_ts = []
mixed_dset_names = []
mixed_dset_classes = []

# pick datasets with more than min_num_req_samples
# then, perform sampling to create a balanced data
for dataname in DEFAULT_LIST_UCR_DSETS:
    skip_this_dataset = False
    print(dataname)

    data_path = os.path.join(DATA_DIR, dataname, f"{dataname}_TRAIN.ts")
    df_train = convert_tsfile_to_dataframe(data_path, return_separate_X_and_y=False)
    tsp = TimeSeriesClassificationPreprocessor(
        input_columns=INPUT_COLUMNS,
        label_column=LABEL_COLUMN,
    )

    tsp.train(df_train)
    df_prep = tsp.preprocess(df_train)
    train_dataset = ClassificationDFDataset(
        df_prep,
        id_columns=[],
        timestamp_column=None,
        input_columns=INPUT_COLUMNS,
        label_column=LABEL_COLUMN,
        context_length=CONTEXT_LENGTH,
        static_categorical_columns=[],
        enable_padding=False,
        full_series=True,
    )
    print(f"Total Train samples: {len(train_dataset)}")

    label_counts = defaultdict(int)
    for sample in train_dataset:
        label = sample["target_values"]
        label_counts[label.item()] += 1

    label_counts = dict(label_counts)
    sorted_label_counts = dict(sorted(label_counts.items()))
    print(dataname, " ", sorted_label_counts)

    for k, v in sorted_label_counts.items():
        if v < min_num_req_samples:
            print(
                f"[NOTE] : {dataname} have less than the required num. of samples {min_num_req_samples} in class {k}, so skipping this dataset"
            )
            skip_this_dataset = True
            break

    if skip_this_dataset:
        continue

    class_indices = defaultdict(list)
    for idx in range(len(train_dataset)):
        label = int(train_dataset[idx]["target_values"])
        # label = int(label)
        class_indices[label].append(idx)

    for class_label, indices in class_indices.items():
        sub_indices = np.random.choice(indices, size=10, replace=False)
        for index in sub_indices:
            datapoint = train_dataset[index]
            datapoint_ts = datapoint["past_values"]  # [T, C]
            datapoint_ts = StandardScaler().fit_transform(datapoint_ts)

            mixed_ts.append(datapoint_ts)
            mixed_dset_names.append(dataname)
            mixed_dset_classes.append(class_label)

mixed_ts, mixed_dset_classes, mixed_dset_names = (
    np.array(mixed_ts),
    np.array(mixed_dset_classes),
    np.array(mixed_dset_names),
)
print(
    "total samples : ",
    mixed_ts.shape,
    mixed_dset_classes.shape,
    mixed_dset_names.shape,
)
np.savez("ucr_for_search.npz", ts=mixed_ts, names=mixed_dset_names, classes=mixed_dset_classes)
