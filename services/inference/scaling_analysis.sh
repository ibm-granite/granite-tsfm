#!/usr/bin/bash
NUM_TIMESERIES=("1000" "500" "250" "125" "60" "30" "15" "10" "5" "1")
NUM_TIMESERIES=("15" "10" "5" "1")
NUM_TARGETS=("100" "90" "80" "70" "60" "50" "40" "30" "20" "10")
#NUM_TIMESERIES=("1")
#NUM_TARGETS=("10")

RUN_TIME=300
PROCESSES=1
SPAWN_RATE=1.0
USERS=1
ONLY_SUMMARY="--only-summary"

for NTS in "${NUM_TIMESERIES[@]}"; do
    echo "******NUM_TIMESERIES=$NTS******"
    for NTARGETS in "${NUM_TARGETS[@]}"; do
        echo "                ******NUM_TARGETS=$NTARGETS******"
        # Necessary so the prometheus metric .db files can wiped out
        # and we can get an accurate per-caller mem/call measure
        oc delete -f examples/k8s/ris3_deployment.yaml
        oc create -f examples/k8s/ris3_deployment.yaml
        # wait for the deployment to be ready
        while ! oc get pods | grep tsfm | grep Running | grep "2/2"; do echo "deployment not ready"; sleep 20; done
        # run locust on the rest-api container
        TSFM_POD=$(oc get pods | grep tsfm | head -n1 | cut -d " " -f1)
        oc exec $TSFM_POD -c rest-client  -- bash -c \
        "NUM_TIMESERIES=$NTS NUM_TARGETS=$NTARGETS locust \
        -f /local-storage/tests/locust/locustfile.py \
        --config /local-storage/tests/locust/locust.ris3.inference.forecast.conf \
        -t $RUN_TIME \
        --headless \
        --users $USERS \
        --processes $PROCESSES \
        --spawn-rate $SPAWN_RATE \
        $ONLY_SUMMARY"
    done # NTARGET
done # NTS