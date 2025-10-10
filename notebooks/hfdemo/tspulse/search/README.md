# Running Search Benchmarking


## Preprocessing Real Data

Run the following commands to download, extract, and preprocess the UCR dataset:
```
curl -OL http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip
unzip Univariate2018_ts.zip
python ucr_preprocessing.py
```

## Setting Up the Environment

If not already installed, add the `faiss` library to your environment:
```
pip install faiss-cpu
```

## Running the Benchmark

Run the following commands:
```
python benchmark.py synth # synthetic dataset
python benchmark.py real  # real dataset
```