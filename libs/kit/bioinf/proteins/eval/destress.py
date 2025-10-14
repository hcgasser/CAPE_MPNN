import os
import pandas as pd


def load_destress_csv(output_dir_path):
    df_destress = pd.read_csv(os.path.join(output_dir_path, 'design_data.csv'))

    # remove the _AF designs if the _exp protein exists
    for exp_design in df_destress[df_destress['design_name'].str.contains('_exp')].design_name:
        exp_design = exp_design.split('_')[0]
        result = df_destress.query(f"design_name == '{exp_design}_AF'")
        if len(result) == 1:
            df_destress = df_destress.drop(result.index[0])

    df_destress.index = df_destress.design_name.apply(lambda x: x.split('_')[0])

    return df_destress