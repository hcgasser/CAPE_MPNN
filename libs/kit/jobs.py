""" This module implements the job management system.

It supports training in various environments - locally (py scripts, notebooks) and on the cluster

Each training process is referred to as a job. A job has a 
unique ID that consists of its environment name, and an
incrementing number (separated by a "_", e.g. mlp_1567951).

A job can resume a previous job.
"""

import os
import matplotlib as mpl
import seaborn as sns
import pandas as pd

import kit.globals as G
from kit.path import join
from kit.data import DD, Split
from kit.data.utils import df_get_highest_in_category


def get_next_job_id(env):
    """returns the next available job ID given a certain environment

    :param environment: name of the environment
    :return: string with the next available job ID
    """

    filename = join(G.ENV.KIT, "current_job_ids", f"{env}.info")

    current_job_id = 0
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            current_job_id = int(file.read())

    next_job_id = current_job_id + 1

    with open(filename, "w", encoding="utf-8") as file:
        file.write(str(next_job_id))

    return f"{env}_{next_job_id}"


def get_job_info(
    job_id,
    metrics=False,
    mhparams=False,
    dhparams=False,
    ckpt=None,
    jobs_directory_path=None,
):
    result = {}
    jobs_directory_path = (
        jobs_directory_path if jobs_directory_path is not None else G.ENV.JOBS
    )
    cont_path = os.path.join(jobs_directory_path, job_id)
    task_file_path = os.path.join(cont_path, "TASK.yaml")
    if not os.path.exists(task_file_path):
        raise FileNotFoundError(f"Job '{job_id}' not found.")
    task = DD.from_yaml(task_file_path)
    df = None
    if (metrics or ckpt is not None) and os.path.exists(
        os.path.join(cont_path, "metrics.csv")
    ):
        df = pd.read_csv(os.path.join(cont_path, "metrics.csv"))
        df["job"] = job_id
    while "JOB_TO_CONTINUE" in task:
        job_to_continue = task.JOB_TO_CONTINUE
        cont_path = os.path.join(G.ENV.JOBS, job_to_continue)
        if (metrics or ckpt is not None) and os.path.exists(
            os.path.join(cont_path, "metrics.csv")
        ):
            df_tmp = pd.read_csv(os.path.join(cont_path, "metrics.csv"))
            df_tmp["job"] = job_to_continue
            df = (
                pd.concat([df_tmp.query(f"step < {df.step.min()}"), df])
                if df is not None
                else df_tmp
            )

        task = DD.from_yaml(os.path.join(cont_path, "TASK.yaml"))

    if mhparams:
        result["mhparams"] = DD.from_yaml(os.path.join(cont_path, "MHPARAMS.yaml"))

    if dhparams:
        result["dhparams"] = DD.from_yaml(os.path.join(cont_path, "DHPARAMS.yaml"))

    if metrics:
        df.set_index("step", inplace=True)
        df.sort_index(inplace=True)
        result["metrics"] = df

    if ckpt:
        ckpt2 = ckpt.split("_")
        ckpt_type, ckpt_metric = ckpt2[0], "_".join(ckpt2[1:])
        if df.index.name != "step":
            df = df.set_index("step")
        df = df.sort_index()
        if ckpt_type == "best":
            ckpt_job_id = df.loc[df[ckpt_metric].idxmin()].job
        elif ckpt_type == "last":
            ckpt_job_id = df.iloc[-1].job

        ckpt_path = os.path.join(
            jobs_directory_path, ckpt_job_id, "checkpoints", f"{ckpt}.ckpt"
        )

        result["ckpt"] = (ckpt_job_id, ckpt_path)

    return result


def plot_job_metrics(
    filename, job_id, x, y, splits=False, xscale="linear", yscale="linear"
):
    df = get_job_info(job_id, metrics=True)["metrics"]
    sns.set(style="ticks")
    _, ax = mpl.pyplot.subplots(figsize=(8, 5))

    # if the y variable is recorded for each split individually
    if splits:
        dfs, palette, mins = [], Split.get_palette(), {}
        for split in Split:
            col = f"{str(split)}_{y}"
            if col in df.columns:
                df_tmp = (
                    df[[col, "epoch"]].query(f"{col}.notna()").rename(columns={col: y})
                )
                df_tmp["split"] = split.long_name()
                if x == "epoch":
                    df_tmp2 = df_get_highest_in_category(
                        df_tmp, "step", "epoch", f"{y}.notna()"
                    )
                    df_tmp = df_tmp2.join(df_tmp, on="step", how="left").drop(
                        columns=["epoch"]
                    )

                min_row = df_tmp.loc[df_tmp[y].idxmin()]
                mins.update({f"C{int(split)*2+1}": (min_row.name, min_row[y])})

                dfs.append(df_tmp)
        df = pd.concat(dfs)
        sns.set_palette("Paired")
        h = sns.lineplot(df, x=x, y=y, hue="split", palette=palette)
        for color, (min_x, min_y) in mins.items():
            mpl.pyplot.scatter(x=[min_x], y=[min_y], c=color)
            text = f"{min_x}: {min_y:.2E}" if min_y < 1 else f"{min_x}: {min_y:.1f}"
            mpl.pyplot.text(x=min_x, y=min_y, s=text, c=color)
    else:
        if x == "epoch":
            df_tmp = df_get_highest_in_category(df, "step", "epoch", f"{y}.notna()")
            df = df_tmp.join(df, on="step", how="left").drop(columns=["epoch"])
        h = sns.lineplot(df, x=x, y=y)
    h.set_xscale(xscale)
    h.set_yscale(yscale)

    if x == "step":
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, pos: f"{x / 1000:,.0f}K")
        )

    mpl.pyplot.savefig(filename)
