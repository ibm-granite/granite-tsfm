from datetime import datetime

import pandas as pd
from tsfminference.dataframe_checks import check
from tsfminference.inference_payloads import ForecastingMetadataInput


def test_check():
    # all good
    df = pd.DataFrame(columns=["date", "id1", "id2", "val"], data=[[datetime.now(), "A", "B", 1.0]])
    mdi = ForecastingMetadataInput(timestamp_column="date", id_columns=["id1", "id2"], target_columns=["val"])
    rc, ex = check(data=df, schema=mdi.model_dump())
    assert rc == 0

    # non-integer type for id
    df = pd.DataFrame(columns=["date", "id1", "id2", "val"], data=[[datetime.now(), "A", 1.0, 1.0]])
    rc, ex = check(data=df, schema=mdi.model_dump())
    assert rc == 1
    assert str(ex).find("id2") >= 0
    # bad column reference
    df = pd.DataFrame(columns=["date", "id1", "id2", "val"], data=[[datetime.now(), "A", "B", 1.0]])
    mdi = ForecastingMetadataInput(timestamp_column="timestamp", id_columns=["id1", "id2"], target_columns=["val"])
    rc, ex = check(data=df, schema=mdi.model_dump())
    assert str(ex).find("timestamp") >= 0
    assert rc == 1
    # bad column reference for id col
    df = pd.DataFrame(columns=["date", "id1", "id2", "val"], data=[[datetime.now(), "A", "B", 1.0]])
    mdi = ForecastingMetadataInput(timestamp_column="date", id_columns=["id1", "foobar"], target_columns=["val"])
    rc, ex = check(data=df, schema=mdi.model_dump())
    assert str(ex).find("foobar") >= 0
    assert rc == 1
    # bad column reference for target col
    df = pd.DataFrame(columns=["date", "id1", "id2", "val"], data=[[datetime.now(), "A", "B", 1.0]])
    mdi = ForecastingMetadataInput(
        timestamp_column="date",
        id_columns=["id1", "id2"],
        target_columns=["foobar"],
    )
    rc, ex = check(data=df, schema=mdi.model_dump())
    assert str(ex).find("foobar") >= 0
    assert rc == 1
    # non-existent column
    df = pd.DataFrame(columns=["date", "id1", "id2", "val"], data=[[datetime.now(), "A", "B", 1.0]])
    mdi = ForecastingMetadataInput(
        timestamp_column="date", id_columns=["id1", "id2"], target_columns=["val"], observable_columns=["foobar"]
    )
    rc, ex = check(data=df, schema=mdi.model_dump())
    assert str(ex).find("foobar") >= 0
    assert rc == 1
    # duplicate specifier - need disjoint sets
    df = pd.DataFrame(columns=["date", "id1", "id2", "val"], data=[[datetime.now(), "A", "B", 1.0]])
    mdi = ForecastingMetadataInput(
        timestamp_column="date", id_columns=["id1", "id2"], target_columns=["val"], observable_columns=["val"]
    )
    rc, ex = check(data=df, schema=mdi.model_dump())
    assert str(ex).find("val") >= 0
    assert rc == 1
    # duplicate specifier - need disjoint sets
    df = pd.DataFrame(columns=["date", "id1", "id2", "val"], data=[[datetime.now(), "A", "B", 1.0]])
    mdi = ForecastingMetadataInput(
        timestamp_column="date", id_columns=["id1", "id2"], target_columns=["val"], observable_columns=["date"]
    )
    rc, ex = check(data=df, schema=mdi.model_dump())
    assert str(ex).find("date") >= 0
    assert rc == 1
