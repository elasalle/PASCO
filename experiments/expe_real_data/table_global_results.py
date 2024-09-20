# in anaconda prompt
# scp elasalle@s92gpu2.cbp.ens-lyon.fr:/projects/users/elasalle/Parallel_Structured_Coarse_Grained_Spectral_Clustering/expes/expes_realdata/result_real_data/res_mag_CSC_MDL_SC_infomap_leiden_louvain_ot.pickle C:\Users\user\Documents\GitHub\Parallel_Structured_Coarse_Grained_Spectral_Clustering\expes\expes_realdata\result_real_data

import pandas as pd
import numpy as np
import argparse
import pickle

def add_midrules(latex_table, solvers):
    # Split the table into lines
    lines = latex_table.split('\n')
    
    # Add \midrule after each solver
    new_lines = []
    for line in lines:
        if line.split(' &')[0] in solvers[1:]:
            new_lines.append("\\midrule")
        new_lines.append(line)
    
    # Join the modified lines into a single string
    return '\n'.join(new_lines)

def sci_format(x):
    # Format number in scientific notation
    formatted = f'{x:.2e}'
    # Replace "+" with "", remove leading zeros after "e"
    base, exponent = formatted.split('e')
    if exponent.startswith('+'):
        exponent = exponent[1:]  # Remove leading "+"
    if exponent.startswith('0'):
        exponent = exponent[1:]
    return f'{base}e{exponent}'
def sub1_format(x):
    return f'{x:4.3}'

def sci_to_float(x):
    """Convert a string in scientific notation back to a float."""
    try:
        return float(x.replace('e', 'E'))
    except ValueError:
        return np.nan
    
def combined_format(x, is_bold, is_boxed):
    """Format the value with both bold and boxed LaTeX commands as necessary."""
    if is_bold and is_boxed:
        return f'\\fbox{{\\textbf{{{x}}}}}'
    elif is_bold:
        return f'\\textbf{{{x}}}'
    elif is_boxed:
        return f'\\fbox{{{x}}}'
    return x

def sci_to_float(x):
    """Convert a string in scientific notation back to a float."""
    try:
        return float(x.replace('e', 'E'))
    except ValueError:
        return np.nan

def precompute_extremal_values(df, perfs):
    """Precompute the overall minimum and maximum values for each performance metric."""
    min_values = {}
    max_values = {}

    for col in perfs:
        numeric_vals = df[col].apply(sci_to_float)
        min_values[col] = numeric_vals.min()
        max_values[col] = numeric_vals.max()

    return min_values, max_values

def apply_combined_formatting(df, solver, perfs, min_values, max_values):
    """Apply both bold and box formatting to extremal values in the DataFrame for each solver."""
    df_solver = df[df['methods'].str.startswith(solver)]

    min_cols = ['time', 'gnCut', 'dl']
    max_cols = ['ami', 'modularity']

    for col in perfs:
        numeric_vals = df_solver[col].apply(sci_to_float)
        if col in min_cols:
            min_value = numeric_vals.min()
            df.loc[df['methods'].str.startswith(solver), col] = \
                df.loc[df['methods'].str.startswith(solver), col].apply(
                    lambda x: combined_format(x, is_bold=sci_to_float(x) == min_value, is_boxed=sci_to_float(x) == min_values[col])
                )
        elif col in max_cols:
            max_value = numeric_vals.max()
            df.loc[df['methods'].str.startswith(solver), col] = \
                df.loc[df['methods'].str.startswith(solver), col].apply(
                    lambda x: combined_format(x, is_bold=sci_to_float(x) == max_value, is_boxed=sci_to_float(x) == max_values[col])
                )

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test on real data. To evaluate time and performance of various methods including pasco.')
    parser.add_argument('-d', '--dataset', nargs='?', type=str,
                        help='dataset on which the experiment is performed', default="arxiv",
                        choices=["arxiv", "mag", "products"])
    args = parser.parse_args()

    dataset = args.dataset
    suffix = "quadratic_ot"
    solvers = ['SC','CSC','louvain','leiden','MDL','infomap']
    res_dir = "results/"
    table_dir = "../data/tables/real_data"
    save = True
    cvxhull = True    
    nt_max = 15
    extra_vmax_coef = 1.05

    sorted_solvers = solvers.copy()
    sorted_solvers.sort()
    saving_file_name = res_dir + 'res_' + dataset + '_' + '_'.join(sorted_solvers) + "_" + suffix +'.pickle' # when using pickle to save the results
    with open(saving_file_name, 'rb') as f:
        results = pickle.load(f)
    # results = np.load(saving_file_name, allow_pickle='TRUE').item()

    # gather parameters
    rhos = list(results[solvers[0]].keys())[1:] # keep only the rhos > 1
    nts = list(results[solvers[0]][rhos[-1]].keys())
    perfs = ["time", "ami", "modularity", "gnCut", "dl"]
    print('rho in ', rhos)
    print('nts in ', nts)

    rho = rhos[0]
    # Assuming the necessary variables (results, solvers, rhos, perfs, nts, etc.) are already defined

    # Collect data into a pandas DataFrame
    data = []

    # Loop through solvers and nts to collect data
    for solver in solvers:
        row = [solver]
        row.extend([results[solver][1][1].get(perf, np.nan) for perf in perfs])
        data.append(row)
        for nt in nts:
            row = [solver+r"+PASCO $(t={})$".format(nt)]
            row.extend([results[solver][rho][nt].get(perf, np.nan) for perf in perfs])
            data.append(row)

    # Create DataFrame
    column_names = ["methods"] + perfs
    df = pd.DataFrame(data, columns=column_names)

    # Apply specific formatting
    for col in ['time', 'dl']:
        df[col] = df[col].apply(sci_format)
    for col in ["ami", "modularity", "gnCut"]:
        df[col] = df[col].apply(sub1_format)

    # Precompute overall extremal values
    min_values, max_values = precompute_extremal_values(df, perfs)

    # Apply combined formatting for extremal values and overall optimal values
    for solver in solvers:
        df = apply_combined_formatting(df, solver, perfs, min_values, max_values)

    # Define the mapping of column names to their LaTeX representations
    column_name_mapping = {
        'ami': r'ami $\uparrow$',
        'modularity': r'modularity $\uparrow$',
        'time': r'time $\downarrow$',
        'gnCut': r'gnCut $\downarrow$',
        'dl': r'dl $\downarrow$'
    }
    # Rename the columns in the DataFrame
    df.rename(columns=column_name_mapping, inplace=True)

    # Convert DataFrame to LaTeX format
    latex_table = df.to_latex(index=False, column_format="l" + "c" * (len(column_names) - 1))
    
    latex_table = add_midrules(latex_table, solvers)

    # Extract only the tabular part
    tabular_start = latex_table.find(r'\begin{tabular}')
    tabular_end = latex_table.find(r'\end{tabular}') + len(r'\end{tabular}')
    tabular_only = latex_table[tabular_start:tabular_end]
    
    # Save LaTeX code to a file
    latex_file_name = table_dir + dataset + "_tabular.tex"
    with open(latex_file_name, 'w') as f:
        f.write(tabular_only)
