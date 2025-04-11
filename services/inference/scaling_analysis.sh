#!/usr/bin/bash

# wait until the service is ready


NUM_TIMESERIES=("1000" "500" "250" "125" "60" "30" "15" "10" "5" "1")
NUM_TIMESERIES=("1000")
RUN_TIME=120
PROCESSES=1
SPAWN_RATE=1.0
USERS=3
#ONLY_SUMMARY="--only-summary"

for NTS in "${NUM_TIMESERIES[@]}"; do

    # Necessary so the prometheus metric .db files can wiped out
    #oc delete -f examples/k8s/ris3_deployment.yaml
    #oc create -f examples/k8s/ris3_deployment.yaml
    # wait for the deployment to be ready
    while ! oc get pods | grep tsfm | grep Running | grep "2/2"; do echo "deployment not ready"; sleep 20; done
    # run locust on the rest-api container
    TSFM_POD=$(oc get pods | grep tsfm | head -n1 | cut -d " " -f1)
    echo "******NUM_TIMESERIES=$NTS******"
    oc exec $TSFM_POD -c rest-client  -- bash -c \
    "NUM_TIMESERIES=$NTS locust \
    -f /local-storage/tests/locust/locustfile.py \
    --config /local-storage/tests/locust/locust.ris3.inference.forecast.conf \
    -t $RUN_TIME \
    --headless \
    --users $USERS \
    --processes $PROCESSES \
    --spawn-rate $SPAWN_RATE \
    $ONLY_SUMMARY"

done