import base64
import io

from services.finetuning.tsfmfinetuning.ioutils import to_pandas

from .payloads import DataInput


async def load_timeseries(input: DataInput) -> pd.DataFrame:
    """Load time series data either from inline arrays or a URI."""

    # we're not reading from a uri
    if input.data is not None:
        # Store data in an I/O buffer
        buf = io.BytesIO()
        buf.write(input.data)
        buf.seek(0)
        output = base64.b64encode(buf.getbuffer())
        return to_pandas(uri=output.decode("utf-8"), timestamp_column=input.timestamp_column)
    elif input.data_uri:
        return to_pandas(uri=input.data_uri, timestamp_column=input.timestamp_column)
        # parsed = urlparse(data.data_uri)
        # scheme = parsed.scheme

        # Local file
        # if scheme in ("file", ""):
        #    df = pd.read_csv(parsed.path)
        #    return pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]))

        # HTTP(S) remote resource
        # elif scheme in ("http", "https"):
        #    async with aiohttp.ClientSession() as session:
        #        async with session.get(data.data_uri) as resp:
        #            if resp.status != 200:
        #                raise ValueError(f"Failed to fetch {data.data_uri}: {resp.status}")
        #            content = await resp.read()
        #            df = pd.read_csv(io.BytesIO(content))
        #            return pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]))

        # else:
        #    raise ValueError(f"Unsupported URI scheme: {scheme}")

    else:
        raise ValueError("No data provided.")
