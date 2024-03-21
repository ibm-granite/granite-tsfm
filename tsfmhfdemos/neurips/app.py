# Copyright contributors to the TSFM project
#
"""Core code for the demo app"""

import logging
import re

import numpy as np
import pandas as pd
import streamlit as st

from tsfmhfdemos.neurips.backends.v1 import GLOBAL_CONFIG, model_util


logging.basicConfig(level=logging.INFO)

# **********************************************************
# There should be no need to edit anything below
# unless you are altering the content of the
# UI itself
# **********************************************************

logging.info(GLOBAL_CONFIG)

MODEL_DOCS = GLOBAL_CONFIG["MODEL_DOCS"]
create_figure = model_util.create_figure
# provides the forecast for a pretrained model
forecast = model_util.forecast

DATASETS = GLOBAL_CONFIG["DATASETS"]
MODELS = GLOBAL_CONFIG["MODELS"]
INFERENCE_APPROACHES = GLOBAL_CONFIG["INFERENCE_APPROACHES"]
METRICS = GLOBAL_CONFIG["INFERENCE_METRICS"]


def tsforecasting_with_fmdls():
    st.set_page_config(
        page_title=GLOBAL_CONFIG["title"],
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(GLOBAL_CONFIG["title"])
    st.write("<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True)

    st.write(GLOBAL_CONFIG["intro"])

    # required_cols = ["ds", "y"]

    with st.sidebar:
        st.subheader("Select the pre-trained model to use.")
        model_name = st.radio(
            "Model:",
            MODELS.keys(),
        )

    with st.sidebar:
        st.subheader("Select which dataset to use for tuning and testing.")
        dataset_name = st.radio(
            "Dataset:",
            DATASETS.keys(),
        )

    with st.sidebar:
        st.subheader("Select the tuning approach.")
        approach_name = st.radio(
            "Tuning Approach:",
            INFERENCE_APPROACHES.keys(),
        )

        dataset_meta = DATASETS[dataset_name]
        model_meta = MODELS[model_name]
        approach_meta = INFERENCE_APPROACHES[approach_name]

    # tabs
    tab_forecast, tab_leaderboard, tab_docs = st.tabs(
        [
            ":chart: Forecast",
            ":runner: Leader Board",
            ":information_source: Documentation",
        ]
    )

    with tab_forecast:
        # ### Dataframe of selected data
        # st.json(GLOBAL_CONFIG)

        st.header(dataset_name)

        # obtain forecast
        # forecasts = model_util.forecast(**dataset_meta, **model_meta)
        # plot forecast and ground true
        # num_cols = 1
        # columns = st.columns(num_cols)

        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            st.subheader("Forecast results")
            for idx, channel in enumerate(dataset_meta["channel_plots"]):
                # col = columns[idx % num_cols]
                st.plotly_chart(
                    model_util.create_figure(**dataset_meta, **model_meta, **approach_meta, channel=channel),
                    use_container_width=True,
                    fig_size=(1600, 200),
                )

        with col2:
            st.subheader("Performance")
            df_perf = model_util.get_performance(metrics=METRICS, **dataset_meta, **model_meta, **approach_meta)

            df_perf_styled = df_perf.style.set_table_styles(
                [
                    {"selector": "th", "props": "background-color: whitesmoke;"},
                ]
            ).format(precision=3)  # .style.hide(axis="index")
            st.write(df_perf_styled.to_html(), unsafe_allow_html=True)
            st.write("")

        # st.dataframe(df_perf)
        # add output of results

        st.subheader("Dataset")
        df = model_util.csv_to_df(dataset_meta)
        st.dataframe(df)

    with tab_leaderboard:
        st.subheader("Current Leader Board")
        # leaderboard = pd.DataFrame(
        #     np.random.randn(len(INFERENCE_APPROACHES), len(METRICS)),
        #     columns=METRICS,
        #     index=INFERENCE_APPROACHES,
        # )

        table_source = r"""
\begin{tabular}{cc|c|cc|cc|cc|cc|cc|ccc}
\cline{2-15}
&\multicolumn{2}{c|}{Models} & \multicolumn{2}{c}{\textbf{\citsm-Best}} & \multicolumn{2}{c|}{DLinear} & \multicolumn{2}{c|}{PatchTST}& \multicolumn{2}{c|}{FEDformer}& \multicolumn{2}{c|}{Autoformer}& \multicolumn{2}{c}{Informer} \\
\cline{2-15}
&\multicolumn{2}{c|}{Metric}&MSE&MAE&MSE&MAE&MSE&MAE&MSE&MAE&MSE&MAE&MSE&MAE\\
\cline{2-15}
&\multirow{4}*{\rotatebox{90}{ETTH1}} & 96 & \textbf{0.368$\pm$0.001} & \textbf{0.398$\pm$0.001} & 0.375 & \uline{0.399} & \uline{0.370} & 0.400 & 0.376 & 0.419 & 0.449 & 0.459 & 0.865 & 0.713 \\
&\multicolumn{1}{c|}{} & 192 & \textbf{0.399$\pm$0.002} & \uline{0.418$\pm$0.001} & \uline{0.405} & \textbf{0.416} & 0.413 & 0.429 & 0.420 & 0.448 & 0.500 & 0.482 & 1.008 & 0.792 \\
&\multicolumn{1}{c|}{} & 336 & \textbf{0.421$\pm$0.004} & \textbf{0.436$\pm$0.003} & 0.439 & 0.443 & \uline{0.422} & \uline{0.440} & 0.459 & 0.465 & 0.521 & 0.496 & 1.107 & 0.809 \\
&\multicolumn{1}{c|}{} & 720 & \textbf{0.444$\pm$0.003} & \textbf{0.467$\pm$0.002} & 0.472 & 0.490 & \uline{0.447} & \uline{0.468} & 0.506 & 0.507 & 0.514 & 0.512 & 1.181 & 0.865 \\
\cline{2-15}
&\multirow{4}*{\rotatebox{90}{ETTH2}} & 96 & \uline{0.276$\pm$0.006} & \textbf{0.337$\pm$0.003} & 0.289 & \uline{0.353} & \textbf{0.274} & \textbf{0.337} & 0.346 & 0.388 & 0.358 & 0.397 & 3.755 & 1.525 \\
&\multicolumn{1}{c|}{} & 192 & \textbf{0.330$\pm$0.003} & \textbf{0.374$\pm$0.001} & 0.383 & 0.418 & \uline{0.341} & \uline{0.382} & 0.429 & 0.439 & 0.456 & 0.452 & 5.602 & 1.931 \\
&\multicolumn{1}{c|}{} & 336 & \uline{0.357$\pm$0.001} & \uline{0.401$\pm$0.002} & 0.448 & 0.465 & \textbf{0.329} & \textbf{0.384} & 0.496 & 0.487 & 0.482 & 0.486 & 4.721 & 1.835 \\
&\multicolumn{1}{c|}{} & 720 & \uline{0.395$\pm$0.003} & \uline{0.436$\pm$0.003} & 0.605 & 0.551 & \textbf{0.379} & \textbf{0.422} & 0.463 & 0.474 & 0.515 & 0.511 & 3.647 & 1.625 \\
\cline{2-15}
&\multirow{4}*{\rotatebox{90}{ETTM1}} & 96 & \textbf{0.291$\pm$0.002} & \uline{0.346$\pm$0.002} & 0.299 & \textbf{0.343} & \uline{0.293} & \uline{0.346} & 0.379 & 0.419 & 0.505 & 0.475 & 0.672 & 0.571 \\
&\multicolumn{1}{c|}{} & 192 & \textbf{0.333$\pm$0.002} & \uline{0.369$\pm$0.002} & \uline{0.335} & \textbf{0.365} & \textbf{0.333} & 0.370 & 0.426 & 0.441 & 0.553 & 0.496 & 0.795 & 0.669 \\
&\multicolumn{1}{c|}{} & 336 & \textbf{0.365$\pm$0.005} & \textbf{0.385$\pm$0.004} & \uline{0.369} & \uline{0.386} & \uline{0.369} & 0.392 & 0.445 & 0.459 & 0.621 & 0.537 & 1.212 & 0.871 \\
&\multicolumn{1}{c|}{} & 720 & \textbf{0.416$\pm$0.002} & \textbf{0.413$\pm$0.001} & \uline{0.425} & 0.421 & \textbf{0.416} & \uline{0.420} & 0.543 & 0.490 & 0.671 & 0.561 & 1.166 & 0.823 \\
\cline{2-15}
&\multirow{4}*{\rotatebox{90}{ETTM2}} & 96 & \textbf{0.164$\pm$0.002} & \textbf{0.255$\pm$0.002} & 0.167 & 0.260 & \uline{0.166} & \uline{0.256} & 0.203 & 0.287 & 0.255 & 0.339 & 0.365 & 0.453 \\
&\multicolumn{1}{c|}{} & 192 & \textbf{0.219$\pm$0.002} & \textbf{0.293$\pm$0.002} & 0.224 & 0.303 & \uline{0.223} & \uline{0.296} & 0.269 & 0.328 & 0.281 & 0.340 & 0.533 & 0.563 \\
&\multicolumn{1}{c|}{} & 336 & \textbf{0.273$\pm$0.003} & \textbf{0.329$\pm$0.003} & 0.281 & \uline{0.342} & \uline{0.274} & \textbf{0.329} & 0.325 & 0.366 & 0.339 & 0.372 & 1.363 & 0.887 \\
&\multicolumn{1}{c|}{} & 720 & \textbf{0.358$\pm$0.002} & \textbf{0.380$\pm$0.001} & 0.397 & 0.421 & \uline{0.362} & \uline{0.385} & 0.421 & 0.415 & 0.433 & 0.432 & 3.379 & 1.338 \\
\cline{2-15}
&\multirow{4}*{\rotatebox{90}{Electricity}} & 96 & \textbf{0.129$\pm$1e-4} & \uline{0.224$\pm$0.001} & \uline{0.140} & 0.237 & \textbf{0.129} & \textbf{0.222} & 0.193 & 0.308 & 0.201 & 0.317 & 0.274 & 0.368 \\
&\multicolumn{1}{c|}{} & 192 & \textbf{0.146$\pm$0.001} & \uline{0.242$\pm$1e-4} & 0.153 & 0.249 & \uline{0.147} & \textbf{0.240} & 0.201 & 0.315 & 0.222 & 0.334 & 0.296 & 0.386 \\
&\multicolumn{1}{c|}{} & 336 & \textbf{0.158$\pm$0.001} & \textbf{0.256$\pm$0.001} & 0.169 & 0.267 & \uline{0.163} & \uline{0.259} & 0.214 & 0.329 & 0.231 & 0.338 & 0.300 & 0.394 \\
&\multicolumn{1}{c|}{} & 720 & \textbf{0.186$\pm$0.001} & \textbf{0.282$\pm$0.001} & 0.203 & 0.301 & \uline{0.197} & \uline{0.290} & 0.246 & 0.355 & 0.254 & 0.361 & 0.373 & 0.439 \\
\cline{2-15}
&\multirow{4}*{\rotatebox{90}{Traffic}} & 96 & \textbf{0.356$\pm$0.002} & \textbf{0.248$\pm$0.002} & 0.410 & 0.282 & \uline{0.360} & \uline{0.249} & 0.587 & 0.366 & 0.613 & 0.388 & 0.719 & 0.391 \\
&\multicolumn{1}{c|}{} & 192 & \textbf{0.377$\pm$0.003} & \uline{0.257$\pm$0.002} & 0.423 & 0.287 & \uline{0.379} & \textbf{0.256} & 0.604 & 0.373 & 0.616 & 0.382 & 0.696 & 0.379 \\
&\multicolumn{1}{c|}{} & 336 & \textbf{0.385$\pm$0.002} & \textbf{0.262$\pm$0.001} & 0.436 & 0.296 & \uline{0.392} & \uline{0.264} & 0.621 & 0.383 & 0.622 & 0.337 & 0.777 & 0.420 \\
&\multicolumn{1}{c|}{} & 720 & \textbf{0.424$\pm$0.001} & \textbf{0.283$\pm$0.001} & 0.466 & 0.315 & \uline{0.432} & \uline{0.286} & 0.626 & 0.382 & 0.660 & 0.408 & 0.864 & 0.472 \\
\cline{2-15}
&\multirow{4}*{\rotatebox{90}{Weather}} & 96 & \textbf{0.146$\pm$0.001} & \textbf{0.197$\pm$0.002} & 0.176 & 0.237 & \uline{0.149} & \uline{0.198} & 0.217 & 0.296 & 0.266 & 0.336 & 0.300 & 0.384 \\
&\multicolumn{1}{c|}{} & 192 & \textbf{0.191$\pm$0.001} & \textbf{0.240$\pm$0.001} & 0.220 & 0.282 & \uline{0.194} & \uline{0.241} & 0.276 & 0.336 & 0.307 & 0.367 & 0.598 & 0.544 \\
&\multicolumn{1}{c|}{} & 336 & \textbf{0.243$\pm$0.001} & \textbf{0.279$\pm$0.002} & 0.265 & 0.319 & \uline{0.245} & \uline{0.282} & 0.339 & 0.380 & 0.359 & 0.395 & 0.578 & 0.523 \\
&\multicolumn{1}{c|}{} & 720 & \uline{0.316$\pm$0.001} & \textbf{0.333$\pm$0.002} & 0.323 & 0.362 & \textbf{0.314} & \uline{0.334} & 0.403 & 0.428 & 0.419 & 0.428 & 1.059 & 0.741 \\
\cline{2-15}
% &\multicolumn{4}{c|}{\makecell{\textbf{\citsm-Best} \textbf{\% improvement}}}& \textbf{8\%} & \textbf{6.8\%}& \textbf{0.7\%} & \textbf{0.4\%} & \textbf{22.9\%} & \textbf{18.2\%} & \textbf{30.1\%} & \textbf{22.7\%} & \textbf{64\%} & \textbf{50.3\%} \\
% &\multicolumn{4}{c|}{\makecell{\textbf{\citsm-Best} \textbf{\% improvement (MSE)}}}& \multicolumn{2}{c}{\textbf{8\%}} & \multicolumn{2}{c}{ \textbf{0.7\%}}  & \multicolumn{2}{c}{\textbf{22.9\%}}  & \multicolumn{2}{c}{\textbf{30.1\%}}  & \multicolumn{2}{c}{\textbf{64\%}}  \\
&\multicolumn{4}{c|}{\makecell{\textbf{\citsm-Best} \textbf{\% improvement (MSE)}}}& \multicolumn{2}{c}{\textbf{8\%}} & \multicolumn{2}{c}{ \textbf{1\%}}  & \multicolumn{2}{c}{\textbf{23\%}}  & \multicolumn{2}{c}{\textbf{30\%}}  & \multicolumn{2}{c}{\textbf{64\%}}  \\
\cline{2-15}
\end{tabular}
"""

        out = re.sub(r"\\textbf{([^&]*)}", r"\1", table_source)
        out = re.sub(r"\\uline{([^&]*)}", r"\1", out)
        out = re.sub(r"\s*|\$\\pm\$[^&]*|\\cline{.*}", "", out)
        vals = np.array([r.split("&")[3:] for r in out.split(r"\\")[2:30]]).astype(float)

        leaderboard = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [
                    [
                        "ETTh1",
                        "ETTh2",
                        "ETTm1",
                        "ETTm2",
                        "Electricity",
                        "Traffic",
                        "Weather",
                    ],
                    [96, 192, 336, 720],
                ],
                names=["Dataset", "Prediction length"],
            ),
            columns=pd.MultiIndex.from_product(
                [
                    [
                        "PatchTSMixer",
                        "DLinear",
                        "PatchTST",
                        "FEDformer",
                        "Autoformer",
                        "Informer",
                    ],
                    ["MSE", "MAE"],
                ],
                names=["Model", "Metric"],
            ),
            data=vals,
        )

        def highlight_best(s, props=""):
            for metric in ["MSE", "MAE"]:
                mask = s.index.droplevel(0) == metric
                s[mask] = s[mask] == np.nanmin(s.values[mask])

            ret = np.where(s, props, "")
            return ret

        def highlight_second_best(s, props=""):
            for metric in ["MSE", "MAE"]:
                mask = s.index.droplevel(0) == metric
                v_ord = np.sort(s.values[mask])
                v_min = v_ord[0]
                v_second = v_ord[1]
                s[mask] = (s[mask] == v_second) & (v_min != v_second)

            ret = np.where(s, props, "")
            return ret

        # (df.style.apply(highlight_max, axis=0, props='background-color:green;', subset=['A','B'])
        #  .apply(highlight_min, axis=0, props='background-color:red;', subset=['A','B'])

        # st.dataframe(leaderboard, hide_index=False)

        leaderboard = leaderboard[
            [
                "PatchTSMixer",
                "PatchTST",
                "DLinear",
                "FEDformer",
                "Autoformer",
                "Informer",
            ]
        ]

        leaderboard_styled = (
            leaderboard.style.set_table_styles(
                [
                    {"selector": "th", "props": "background-color: whitesmoke;"},
                ]
            )
            .format(precision=3)
            .apply(highlight_best, axis=1, props="font-weight: bold;")
            .apply(highlight_second_best, axis=1, props="text-decoration: underline;")
        )

        st.write(leaderboard_styled.to_html(), unsafe_allow_html=True)
        # st.dataframe(leaderboard_styled)

        st.write("Source: https://arxiv.org/abs/2306.09364")

    with tab_docs:
        # drop [0] below if more than one tab
        tab_desc = st.tabs(["Description of the models"])[0]

        with tab_desc:
            for model_name in MODELS.keys():
                st.header(model_name)
                model_card_name = MODELS[model_name]["card"]
                st.subheader("Abstract")
                st.write(f"""{MODEL_DOCS[model_card_name]['Abstract']}""")
                st.subheader("Model Architecture")
                st.image(**MODEL_DOCS[model_card_name]["figure"])
                st.write(f"""{MODEL_DOCS[model_card_name]['Model Architecture']}""")
                # st.subheader("Secondary use")
                # st.write(f"""{MODEL_DOCS[model_card_name]['Secondary use']}""")
                # st.subheader("Limitations")
                # st.write(f"""{MODEL_DOCS[model_card_name]['Limitations']}""")
                # st.subheader("Training data")
                # st.write(f"""{MODEL_DOCS[model_card_name]['Training data']}""")
                st.subheader("BibTex/Citation Info")
                st.code(f"""{MODEL_DOCS[model_card_name]['Citation Info']}""")


if __name__ == "__main__":
    tsforecasting_with_fmdls()
