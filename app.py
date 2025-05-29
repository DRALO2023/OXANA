import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind # Ensure this is imported
import zipfile
import os
import io
# import base64 # Removed as unused
from datetime import datetime # Keep for now, pd.Timedelta might need it implicitly or future use.
# from IPython.display import display # Note: This might need to be removed or handled differently in Streamlit # Removed
# import ipywidgets as widgets # Note: This will be replaced with Streamlit widgets # Removed
# from google.colab import files # This will be removed # Removed

# Helper Functions
def significance_stars(p):
    if p is None or pd.isna(p): return '' 
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

def cohen_d(x, y):
    x = x.dropna()
    y = y.dropna()
    if len(x) < 2 or len(y) < 2: 
        return np.nan
        
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof < 1: 
        return np.nan 
    
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    var_x = max(0, var_x if not pd.isna(var_x) else 0)
    var_y = max(0, var_y if not pd.isna(var_y) else 0)
    
    pooled_std_numerator = ((nx - 1) * var_x) + ((ny - 1) * var_y)

    if dof == 0: 
        return 0 if pooled_std_numerator == 0 and np.mean(x) == np.mean(y) else np.nan

    pooled_std = np.sqrt(pooled_std_numerator / dof)
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    if pd.isna(pooled_std) or pd.isna(mean_x) or pd.isna(mean_y): 
        return np.nan

    if pooled_std == 0:
        return 0 if mean_x == mean_y else np.nan 
    
    return (mean_x - mean_y) / pooled_std

def dfs_to_excel_bytes(dfs_map: dict):
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sheet_name, df in dfs_map.items():
                if isinstance(df, pd.DataFrame):
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        processed_data = output.getvalue()
        return processed_data
    except Exception as e:
        st.error(f"Error creating Excel file: {e}")
        return None

def fig_to_bytes(fig):
    if fig is None:
        return None
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        st.error(f"Error converting figure to bytes: {e}")
        return None
    finally:
        if fig:
             plt.close(fig) # Ensure figure is closed even if savefig fails

def create_zip_from_plot_bytes(plot_files_map: dict):
    if not plot_files_map:
        return None
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, plot_bytes in plot_files_map.items():
                if plot_bytes:
                    zf.writestr(filename, plot_bytes)
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    except Exception as e:
        st.error(f"Error creating ZIP file from plots: {e}")
        return None

# Initialize session state
if 'data_processed' not in st.session_state:
    st.session_state['data_processed'] = False
if 'dark_analysis_done' not in st.session_state:
    st.session_state['dark_analysis_done'] = False
if 'dark_analysis_results' not in st.session_state:
    st.session_state['dark_analysis_results'] = None
if 'dark_phase_plots' not in st.session_state:
    st.session_state['dark_phase_plots'] = None
if 'light_analysis_done' not in st.session_state:
    st.session_state['light_analysis_done'] = False
if 'light_analysis_results' not in st.session_state:
    st.session_state['light_analysis_results'] = None
if 'light_phase_plots' not in st.session_state:
    st.session_state['light_phase_plots'] = None
if 'anova_analysis_done' not in st.session_state:
    st.session_state['anova_analysis_done'] = False
if 'anova_results_df' not in st.session_state:
    st.session_state['anova_results_df'] = None
if 'anova_interaction_plot_figures' not in st.session_state:
    st.session_state['anova_interaction_plot_figures'] = None

# Analysis and Plotting Functions for Dark Phase
# def perform_dark_phase_analysis(df_dark_input, vars_to_analyze=['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT']):
#     if df_dark_input is None or df_dark_input.empty:
#         return None 
    
#     df_dark = df_dark_input.copy()
#     print(df_dark.head)
#     print(df_dark.columns)

#     essential_cols = ['CHAN', 'Date', 'Genotype', 'FEED1']
#     for col in essential_cols:
#         if col not in df_dark.columns:
#             st.error(f"Dark Phase Analysis: Essential column '{col}' is missing. Cannot proceed.")
#             return None 
    
#     current_vars_to_analyze = [var for var in vars_to_analyze if var in df_dark.columns]
#     if not current_vars_to_analyze:
#         st.error("Dark Phase Analysis: None of the specified variables for analysis are present in the data.")
#         return None 
        
#     try:
#         daily_feed_sum = df_dark.groupby(['CHAN', 'Date', 'Genotype'])['FEED1'].sum().reset_index(name='Daily_Total_FEED1')
#         daily_feed_sum = daily_feed_sum.sort_values(['CHAN', 'Date'])
#     except KeyError as e:
#         st.error(f"Dark Phase Analysis: Error grouping for daily feed. Missing column: {e}")
#         return None

#     if 'CHAN' in daily_feed_sum.columns and 'Daily_Total_FEED1' in daily_feed_sum.columns:
#         daily_feed_sum['Cumulative_Daily_FEED1'] = daily_feed_sum.groupby('CHAN')['Daily_Total_FEED1'].cumsum()
#     else:
#         daily_feed_sum['Cumulative_Daily_FEED1'] = np.nan


#     avg_daily_feed = daily_feed_sum.groupby(['CHAN', 'Genotype'])['Daily_Total_FEED1'].mean().reset_index(name='Avg_Daily_FEED1')
#     import numpy as np
#     import pandas as pd
#     summary_list = []
#     df_dark_wt_ko = df_dark[df_dark['Genotype'].isin(['WT', 'KO'])]
#     print("df_dark_wt_ko preview:")
#     print(df_dark_wt_ko.head())

#     if df_dark_wt_ko.empty:
#         st.warning("Dark Phase Analysis: No data found for WT or KO genotypes.")

#     for var in current_vars_to_analyze:
#         print(f"\n--- Analyzing variable: '{var}' ---")
        
#         if var not in df_dark_wt_ko.columns:
#             print(f"⚠️ Skipping '{var}' — not found in dataframe columns.")
#             continue

#         for genotype_to_check in ['WT', 'KO']: 
#             print(f"Processing genotype: {genotype_to_check}")
            
#             if genotype_to_check in df_dark_wt_ko['Genotype'].unique():
#                 raw_data = df_dark_wt_ko[df_dark_wt_ko['Genotype'] == genotype_to_check][var]
#                 print(f"Raw data (first 5 rows) for {var} - {genotype_to_check}:\n{raw_data.head().tolist()}")

#                 data = pd.to_numeric(raw_data, errors='coerce').dropna()
#                 print(f"Cleaned numeric data (first 5 rows) for {var} - {genotype_to_check}:\n{data.head().tolist()}")
#                 print(f"Number of valid numeric entries: {len(data)}")

#                 if not data.empty:
#                     mean_val = data.mean()
#                     sem_val = data.sem()
#                     print(f"✅ Mean: {mean_val:.4f}, SEM: {sem_val:.4f}")
#                     summary_list.append({
#                         'variable': var,
#                         'Genotype': genotype_to_check,
#                         'mean': mean_val,
#                         'sem': sem_val
#                     })
#                 else:
#                     print(f"⚠️ No valid numeric data for {var} - {genotype_to_check}.")
#                     summary_list.append({
#                         'variable': var,
#                         'Genotype': genotype_to_check,
#                         'mean': np.nan,
#                         'sem': np.nan
#                     })
#             else:
#                 print(f"⚠️ Genotype '{genotype_to_check}' not found in data.")
#                 summary_list.append({
#                     'variable': var,
#                     'Genotype': genotype_to_check,
#                     'mean': np.nan,
#                     'sem': np.nan
#                 })

#     print("\n✅ Summary List:")
#     for entry in summary_list:
#         print(entry)

#     summary_df = pd.DataFrame(summary_list)
#     print("\n✅ Summary DataFrame Preview:")
#     print(summary_df.head())

#     ttest_results = []

#     print("🔍 Starting T-test analysis...")

#     # Check for WT and KO in df_dark_wt_ko
#     if not df_dark_wt_ko.empty and set(['WT', 'KO']).issubset(df_dark_wt_ko['Genotype'].unique()):
#         print("✅ Dark phase data contains both WT and KO genotypes.")
        
#         for var in current_vars_to_analyze:
#             print(f"\nAnalyzing variable: {var}")

#             data_wt = pd.to_numeric(df_dark_wt_ko[df_dark_wt_ko['Genotype'] == 'WT'][var], errors='coerce').dropna()
#             data_ko = pd.to_numeric(df_dark_wt_ko[df_dark_wt_ko['Genotype'] == 'KO'][var], errors='coerce').dropna()
            
#             print(f"WT sample size: {len(data_wt)}, KO sample size: {len(data_ko)}")

#             if len(data_wt) >= 2 and len(data_ko) >= 2:
#                 try:
#                     t_stat, p_val = ttest_ind(data_wt, data_ko, equal_var=False, nan_policy='omit')
#                     d = cohen_d(data_wt, data_ko)
#                     star = significance_stars(p_val)
#                     print(f"T-test success: t={t_stat:.3f}, p={p_val:.4g}, Cohen's d={d:.3f}, significance={star}")
#                     ttest_results.append({
#                         'variable': var,
#                         't_stat': t_stat,
#                         'p_value': p_val,
#                         'significance': star,
#                         'cohen_d': d
#                     })
#                 except Exception as e:
#                     st.warning(f"T-test failed for variable {var} in dark phase: {e}")
#                     print(f"⚠️ T-test failed for {var}: {e}")
#                     ttest_results.append({'variable': var, 't_stat': np.nan, 'p_value': np.nan, 'significance': 'error', "cohen_d": np.nan})
#             else:
#                 print(f"⚠️ Not enough data for variable {var} (WT: {len(data_wt)}, KO: {len(data_ko)}). Skipping.")

#     # T-test on average daily feed
#     if not avg_daily_feed.empty and 'Genotype' in avg_daily_feed.columns and 'Avg_Daily_FEED1' in avg_daily_feed.columns:
#         avg_daily_feed_wt_ko = avg_daily_feed[avg_daily_feed['Genotype'].isin(['WT', 'KO'])]
#         if set(['WT', 'KO']).issubset(avg_daily_feed_wt_ko['Genotype'].unique()):
#             avg_feed_wt = pd.to_numeric(avg_daily_feed_wt_ko[avg_daily_feed_wt_ko['Genotype'] == 'WT']['Avg_Daily_FEED1'], errors='coerce').dropna()
#             avg_feed_ko = pd.to_numeric(avg_daily_feed_wt_ko[avg_daily_feed_wt_ko['Genotype'] == 'KO']['Avg_Daily_FEED1'], errors='coerce').dropna()

#             print(f"\nAnalyzing Avg_Daily_FEED1 - WT: {len(avg_feed_wt)}, KO: {len(avg_feed_ko)}")

#             if len(avg_feed_wt) >= 2 and len(avg_feed_ko) >= 2:
#                 try:
#                     t_stat_feed, p_val_feed = ttest_ind(avg_feed_wt, avg_feed_ko, equal_var=False, nan_policy='omit')
#                     d_feed = cohen_d(avg_feed_wt, avg_feed_ko)
#                     star_feed = significance_stars(p_val_feed)
#                     print(f"Avg_Daily_FEED1 t={t_stat_feed:.3f}, p={p_val_feed:.4g}, d={d_feed:.3f}, sig={star_feed}")
#                     ttest_results.append({
#                         'variable': 'Avg_Daily_FEED1',
#                         't_stat': t_stat_feed,
#                         'p_value': p_val_feed,
#                         'significance': star_feed,
#                         'cohen_d': d_feed
#                     })
#                 except Exception as e:
#                     st.warning(f"T-test failed for Avg_Daily_FEED1 in dark phase: {e}")
#                     print(f"⚠️ T-test failed for Avg_Daily_FEED1: {e}")
#                     ttest_results.append({'variable': 'Avg_Daily_FEED1', 't_stat': np.nan, 'p_value': np.nan, 'significance': 'error', "cohen_d": np.nan})
#             else:
#                 print("⚠️ Not enough data for Avg_Daily_FEED1. Skipping.")

#     else:
#         print("⚠️ avg_daily_feed is missing required columns or is empty.")

#     # Convert results to DataFrame
#     ttest_df = pd.DataFrame(ttest_results)
#     print("✅ T-test analysis complete.")

#     # Outlier Detection
#     if not avg_daily_feed.empty and 'Avg_Daily_FEED1' in avg_daily_feed.columns:
#         Q1 = avg_daily_feed['Avg_Daily_FEED1'].quantile(0.25)
#         Q3 = avg_daily_feed['Avg_Daily_FEED1'].quantile(0.75)
#         IQR = Q3 - Q1
#         if pd.isna(IQR) or IQR == 0:
#             avg_daily_feed['Outlier'] = False
#             print("⚠️ IQR is 0 or NaN, marking all as non-outliers.")
#         else:
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
#             avg_daily_feed['Outlier'] = (avg_daily_feed['Avg_Daily_FEED1'] < lower_bound) | (avg_daily_feed['Avg_Daily_FEED1'] > upper_bound)
#             outlier_count = avg_daily_feed['Outlier'].sum()
#             print(f"✅ Outlier detection complete. {outlier_count} outliers found.")
#     elif not avg_daily_feed.empty:
#         avg_daily_feed['Outlier'] = False
#         print("⚠️ Avg_Daily_FEED1 column not found, marking all as non-outliers.")

#     # Return the analysis results
#     return {
#         'summary_df': summary_df,
#         'ttest_df': ttest_df,
#         'avg_daily_feed': avg_daily_feed,
#         'cumulative_feed_data_for_plot': daily_feed_sum
#     }
def perform_dark_phase_analysis(df_dark_input, vars_to_analyze=['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT']):
    import pandas as pd
    import numpy as np
    from scipy.stats import ttest_ind

    if df_dark_input is None or df_dark_input.empty:
        return None 
    
    df_dark = df_dark_input.copy()
    
    essential_cols = ['CHAN', 'Date', 'Genotype', 'FEED1']
    for col in essential_cols:
        if col not in df_dark.columns:
            st.error(f"Dark Phase Analysis: Essential column '{col}' is missing. Cannot proceed.")
            return None 

    # Filter only the necessary columns to avoid unnecessary memory usage
    current_vars_to_analyze = [var for var in vars_to_analyze if var in df_dark.columns]
    if not current_vars_to_analyze:
        st.error("Dark Phase Analysis: None of the specified variables for analysis are present in the data.")
        return None 
    # Convert vars to numeric before grouping
    for var in current_vars_to_analyze:
        if var in df_dark.columns:
            df_dark[var] = pd.to_numeric(df_dark[var], errors='coerce')
    # Step 1: Compute daily averages per animal (CHAN)
    daily_avg = df_dark.groupby(['CHAN', 'Date', 'Genotype'])[current_vars_to_analyze].mean().reset_index()

    # Also compute daily total feed
    daily_feed_sum = df_dark.groupby(['CHAN', 'Date', 'Genotype'])['FEED1'].sum().reset_index(name='Daily_Total_FEED1')
    daily_feed_sum = daily_feed_sum.sort_values(['CHAN', 'Date'])
    daily_feed_sum['Cumulative_Daily_FEED1'] = daily_feed_sum.groupby('CHAN')['Daily_Total_FEED1'].cumsum()

    # Merge feed data with variable averages
    daily_avg = pd.merge(daily_avg, daily_feed_sum, on=['CHAN', 'Date', 'Genotype'], how='left')

    # Compute average daily feed per genotype
    avg_daily_feed = daily_feed_sum.groupby(['CHAN', 'Genotype'])['Daily_Total_FEED1'].mean().reset_index(name='Avg_Daily_FEED1')

    # Begin summary and t-test analysis
    summary_list = []
    df_dark_wt_ko = daily_avg[daily_avg['Genotype'].isin(['WT', 'KO'])]

    for var in current_vars_to_analyze:
        for genotype_to_check in ['WT', 'KO']:
            data = df_dark_wt_ko[df_dark_wt_ko['Genotype'] == genotype_to_check][var]
            data = pd.to_numeric(data, errors='coerce').dropna()

            if not data.empty:
                summary_list.append({
                    'variable': var,
                    'Genotype': genotype_to_check,
                    'mean': data.mean(),
                    'sem': data.sem()
                })
            else:
                summary_list.append({
                    'variable': var,
                    'Genotype': genotype_to_check,
                    'mean': np.nan,
                    'sem': np.nan
                })

    summary_df = pd.DataFrame(summary_list)

    # T-test results
    ttest_results = []
    for var in current_vars_to_analyze:
        data_wt = df_dark_wt_ko[df_dark_wt_ko['Genotype'] == 'WT'][var].dropna()
        data_ko = df_dark_wt_ko[df_dark_wt_ko['Genotype'] == 'KO'][var].dropna()
        if len(data_wt) >= 2 and len(data_ko) >= 2:
            t_stat, p_val = ttest_ind(data_wt, data_ko, equal_var=False, nan_policy='omit')
            d = cohen_d(data_wt, data_ko)
            star = significance_stars(p_val)
            ttest_results.append({
                'variable': var,
                't_stat': t_stat,
                'p_value': p_val,
                'significance': star,
                'cohen_d': d
            })

    # Do the same for Avg_Daily_FEED1
    avg_daily_feed_wt_ko = avg_daily_feed[avg_daily_feed['Genotype'].isin(['WT', 'KO'])]
    if set(['WT', 'KO']).issubset(avg_daily_feed_wt_ko['Genotype'].unique()):
        avg_feed_wt = avg_daily_feed_wt_ko[avg_daily_feed_wt_ko['Genotype'] == 'WT']['Avg_Daily_FEED1']
        avg_feed_ko = avg_daily_feed_wt_ko[avg_daily_feed_wt_ko['Genotype'] == 'KO']['Avg_Daily_FEED1']
        if len(avg_feed_wt) >= 2 and len(avg_feed_ko) >= 2:
            t_stat_feed, p_val_feed = ttest_ind(avg_feed_wt, avg_feed_ko, equal_var=False, nan_policy='omit')
            d_feed = cohen_d(avg_feed_wt, avg_feed_ko)
            star_feed = significance_stars(p_val_feed)
            ttest_results.append({
                'variable': 'Avg_Daily_FEED1',
                't_stat': t_stat_feed,
                'p_value': p_val_feed,
                'significance': star_feed,
                'cohen_d': d_feed
            })

    ttest_df = pd.DataFrame(ttest_results)

    # Outlier detection remains the same
    Q1 = avg_daily_feed['Avg_Daily_FEED1'].quantile(0.25)
    Q3 = avg_daily_feed['Avg_Daily_FEED1'].quantile(0.75)
    IQR = Q3 - Q1
    if pd.isna(IQR) or IQR == 0:
        avg_daily_feed['Outlier'] = False
    else:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        avg_daily_feed['Outlier'] = (avg_daily_feed['Avg_Daily_FEED1'] < lower_bound) | (avg_daily_feed['Avg_Daily_FEED1'] > upper_bound)

    return {
        'summary_df': summary_df,
        'ttest_df': ttest_df,
        'avg_daily_feed': avg_daily_feed,
        'cumulative_feed_data_for_plot': daily_feed_sum
    }


def plot_dark_phase_analysis(df_dark_input_for_plot, analysis_results_for_plot, vars_to_analyze_for_plot=['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT']):
    plots = {} 
    if not analysis_results_for_plot: return plots 
    if df_dark_input_for_plot is None or df_dark_input_for_plot.empty: return plots

    summary_df = analysis_results_for_plot.get('summary_df')
    ttest_df = analysis_results_for_plot.get('ttest_df')
    avg_daily_feed = analysis_results_for_plot.get('avg_daily_feed')
    cumulative_feed_data_for_plot = analysis_results_for_plot.get('cumulative_feed_data_for_plot')
    
    actual_vars_in_summary = []
    if summary_df is not None and 'variable' in summary_df.columns:
        for var_check in vars_to_analyze_for_plot:
            if var_check in summary_df['variable'].unique():
                 if not summary_df[(summary_df['variable']==var_check) & (summary_df['mean'].notna())].empty:
                      actual_vars_in_summary.append(var_check)

    if 'Genotype' not in df_dark_input_for_plot.columns: return plots
    df_dark_wt_ko_plot = df_dark_input_for_plot[df_dark_input_for_plot['Genotype'].isin(['WT', 'KO'])]
    if df_dark_wt_ko_plot.empty: return plots

    genotypes_sorted = ['WT', 'KO']
    if not genotypes_sorted : return plots 
    x_pos = np.arange(len(genotypes_sorted))

    fig1, axes = plt.subplots(2, 3, figsize=(18, 11)) 
    axes = axes.flatten() 
    
    plot_count = 0
    for i, var in enumerate(actual_vars_in_summary):
        ax = axes[i]
        plot_count += 1
        if summary_df is None or not all(col in summary_df.columns for col in ['variable', 'Genotype', 'mean', 'sem']):
            ax.text(0.5, 0.5, f"Summary data for {var} error.", ha='center', va='center', transform=ax.transAxes)
            continue

        data_plot = summary_df[(summary_df['variable'] == var) & (summary_df['Genotype'].isin(genotypes_sorted))].set_index('Genotype').reindex(genotypes_sorted)
        
        if not data_plot.empty and not data_plot['mean'].isnull().all(): 
            means = pd.to_numeric(data_plot['mean'].values, errors='coerce')
            sems = pd.to_numeric(data_plot['sem'].values, errors='coerce')
            valid_plot_indices = ~np.isnan(means)
            
            if not np.any(valid_plot_indices): 
                 ax.text(0.5, 0.5, f"Numeric data for {var} error.", ha='center', va='center', transform=ax.transAxes)
                 continue

            bar_colors = ['#66c2a5' if g == 'WT' else '#fc8d62' for g in np.array(genotypes_sorted)[valid_plot_indices]]
            try:
                ax.bar(x_pos[valid_plot_indices], means[valid_plot_indices], yerr=sems[valid_plot_indices] if np.any(valid_plot_indices) else None, capsize=5, color=bar_colors, ecolor='dimgray', error_kw={'capthick':1.5, 'elinewidth':1.5})
                ax.set_xticks(x_pos) 
                ax.set_xticklabels(genotypes_sorted) 
                ax.set_title(f'{var} (Mean ± SEM)')
                ax.set_ylabel(var)
                
                if ttest_df is not None and not ttest_df.empty and 'variable' in ttest_df.columns and var in ttest_df['variable'].values and len(genotypes_sorted) == 2: 
                    pval_row = ttest_df[ttest_df['variable'] == var]
                    if not pval_row.empty:
                        star = pval_row['significance'].iloc[0]
                        if star != 'ns':
                            current_means = means[valid_plot_indices]
                            current_sems = sems[valid_plot_indices] if np.any(valid_plot_indices) else np.zeros_like(current_means)
                            current_sems = np.nan_to_num(current_sems) 

                            if len(current_means) > 0:
                                y_top = (current_means + current_sems).max()
                                y_min_val = current_means.min()
                                line_y = y_top * 1.05 if y_top >= 0 else y_top * 0.95 
                                if y_top == y_min_val : line_y = y_top + 0.05 * abs(y_top) if y_top !=0 else 0.1 
                                ax.plot(x_pos, [line_y, line_y], color='black', lw=1) 
                                ax.text(np.mean(x_pos), line_y, star, ha='center', va='bottom', fontsize=14)
            except Exception as e:
                ax.text(0.5, 0.5, f"Plotting error for {var}: {e}", ha='center', va='center', transform=ax.transAxes, wrap=True)
                st.warning(f"Could not generate bar plot for {var} in dark phase: {e}")
        else:
            ax.text(0.5, 0.5, f"Data for {var} is empty/NaN.", ha='center', va='center', transform=ax.transAxes)

    feed_plot_ax_idx = plot_count 
    if feed_plot_ax_idx < len(axes): 
        ax = axes[feed_plot_ax_idx]
        plot_count +=1
        if avg_daily_feed is not None and not avg_daily_feed.empty and \
           'Genotype' in avg_daily_feed.columns and 'Avg_Daily_FEED1' in avg_daily_feed.columns:
            
            feed_summary_plot = avg_daily_feed[avg_daily_feed['Genotype'].isin(genotypes_sorted)].groupby('Genotype')['Avg_Daily_FEED1'].agg(['mean', 'sem']).reindex(genotypes_sorted)
            
            if not feed_summary_plot.empty and 'mean' in feed_summary_plot.columns and 'sem' in feed_summary_plot.columns and \
               not feed_summary_plot['mean'].isnull().all():
                
                means = pd.to_numeric(feed_summary_plot['mean'].values, errors='coerce')
                sems = pd.to_numeric(feed_summary_plot['sem'].values, errors='coerce')
                valid_plot_indices = ~np.isnan(means)

                if not np.any(valid_plot_indices):
                    ax.text(0.5, 0.5, "Numeric Avg Daily FEED1 error.", ha='center', va='center', transform=ax.transAxes)
                else:
                    try:
                        bar_colors = ['#66c2a5' if g == 'WT' else '#fc8d62' for g in np.array(genotypes_sorted)[valid_plot_indices]]
                        ax.bar(x_pos[valid_plot_indices], means[valid_plot_indices], yerr=sems[valid_plot_indices] if np.any(valid_plot_indices) else None, capsize=5, color=bar_colors, ecolor='dimgray', error_kw={'capthick':1.5, 'elinewidth':1.5})
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(genotypes_sorted)
                        ax.set_title('Avg Daily FEED1 (Mean ± SEM)')
                        ax.set_ylabel('g (average daily intake)')

                        if ttest_df is not None and not ttest_df.empty and 'variable' in ttest_df.columns and \
                           'Avg_Daily_FEED1' in ttest_df['variable'].values and len(genotypes_sorted) == 2:
                            pval_row = ttest_df[ttest_df['variable'] == 'Avg_Daily_FEED1']
                            if not pval_row.empty:
                                star = pval_row['significance'].iloc[0]
                                if star != 'ns':
                                    current_means = means[valid_plot_indices]
                                    current_sems = sems[valid_plot_indices] if np.any(valid_plot_indices) else np.zeros_like(current_means)
                                    current_sems = np.nan_to_num(current_sems)
                                    if len(current_means) > 0:
                                        y_top = (current_means + current_sems).max()
                                        y_min_val = current_means.min()
                                        line_y = y_top * 1.05 if y_top >= 0 else y_top * 0.95
                                        if y_top == y_min_val : line_y = y_top + 0.05 * abs(y_top) if y_top !=0 else 0.1
                                        ax.plot(x_pos, [line_y, line_y], color='black', lw=1)
                                        ax.text(np.mean(x_pos), line_y, star, ha='center', va='bottom', fontsize=14)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Plotting error for Avg FEED1: {e}", ha='center', va='center', transform=ax.transAxes, wrap=True)
                        st.warning(f"Could not generate bar plot for Avg_Daily_FEED1 in dark phase: {e}")
            else:
                ax.text(0.5, 0.5, "Avg Daily FEED1 summary error.", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Avg daily feed raw data error.", ha='center', va='center', transform=ax.transAxes)
    
    for i in range(plot_count, len(axes)): # Remove unused subplots
        if i < len(axes) and fig1.axes[i] is not None : # Check if axes exists before trying to delete
            try:
                fig1.delaxes(axes[i])
            except Exception as e: # Should not happen with check, but for safety
                 st.warning(f"Minor issue cleaning up dark phase plot axes: {e}")


    fig1.suptitle('Mean ± SEM by Genotype during Dark Phase', fontsize=17, y=1.00) 
    fig1.tight_layout(rect=[0, 0.03, 1, 0.96]) 
    plots['mean_sem_plot'] = fig1

    fig2, ax2 = plt.subplots(figsize=(13, 7)) 
    try:
        if cumulative_feed_data_for_plot is not None and not cumulative_feed_data_for_plot.empty and \
           all(col in cumulative_feed_data_for_plot.columns for col in ['Genotype', 'Date', 'Cumulative_Daily_FEED1']) and \
           not cumulative_feed_data_for_plot['Cumulative_Daily_FEED1'].isnull().all() :
            
            df_for_cum_plot = cumulative_feed_data_for_plot.copy()
            df_for_cum_plot['Date'] = pd.to_datetime(df_for_cum_plot['Date'], errors='coerce')
            df_for_cum_plot.dropna(subset=['Date', 'Cumulative_Daily_FEED1'], inplace=True) 
            
            df_for_cum_plot_wt_ko = df_for_cum_plot[df_for_cum_plot['Genotype'].isin(['WT', 'KO'])]

            if not df_for_cum_plot_wt_ko.empty:
                for genotype, color in zip(['WT', 'KO'], ['#66c2a5', '#fc8d62']):
                    if genotype in df_for_cum_plot_wt_ko['Genotype'].unique():
                        plot_data = df_for_cum_plot_wt_ko[df_for_cum_plot_wt_ko['Genotype'] == genotype]
                        plot_data = plot_data.sort_values(by='Date') 
                        
                        mean_cum = plot_data.groupby('Date')['Cumulative_Daily_FEED1'].agg(['mean', 'sem']).reset_index()

                        if not mean_cum.empty and 'mean' in mean_cum.columns and 'sem' in mean_cum.columns and not mean_cum['mean'].isnull().all():
                            mean_val = pd.to_numeric(mean_cum['mean'], errors='coerce')
                            sem_val = pd.to_numeric(mean_cum['sem'], errors='coerce')
                            
                            valid_indices_cum = ~mean_val.isnull()
                            if not np.any(valid_indices_cum): continue 

                            ax2.plot(mean_cum['Date'][valid_indices_cum], mean_val[valid_indices_cum], label=genotype, color=color, marker='o', linestyle='-', markersize=5, linewidth=1.5)
                            sem_val_filled = sem_val[valid_indices_cum].fillna(0) 
                            ax2.fill_between(mean_cum['Date'][valid_indices_cum], 
                                             mean_val[valid_indices_cum] - sem_val_filled, 
                                             mean_val[valid_indices_cum] + sem_val_filled, 
                                             color=color, alpha=0.15) 
                ax2.set_xlabel('Date', fontsize=12)
                ax2.set_ylabel('Mean Cumulative Daily Food Intake (g)', fontsize=12)
                ax2.set_title('Mean Cumulative Daily Food Intake Over Dark Days', fontsize=14)
                if ax2.has_data(): 
                     ax2.legend(title='Genotype', fontsize=10)
                ax2.tick_params(axis='x', rotation=30, labelsize=10) 
                ax2.tick_params(axis='y', labelsize=10)
                ax2.grid(True, linestyle='--', alpha=0.5)
            else: 
                ax2.text(0.5, 0.5, "WT/KO cumulative feed data error for dark phase plot.", ha='center', va='center', transform=ax2.transAxes, wrap=True)
        else: 
            ax2.text(0.5, 0.5, "Cumulative feed raw data error for dark phase plot.", ha='center', va='center', transform=ax2.transAxes, wrap=True)
    except Exception as e:
        ax2.text(0.5, 0.5, f"Cumulative plot error (Dark): {e}", ha='center', va='center', transform=ax2.transAxes, wrap=True)
        st.warning(f"Could not generate cumulative feed plot for dark phase: {e}")

    fig2.tight_layout()
    plots['cumulative_feed_plot'] = fig2
    
    return plots

# Analysis and Plotting Functions for Light Phase
# def perform_light_phase_analysis(df_light_input, vars_to_analyze=['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT']):
#     if df_light_input is None or df_light_input.empty:
#         return None
    
#     df_light = df_light_input.copy()
#     print(df_light.head())
#     print(df_light.columns)

#     essential_cols = ['CHAN', 'Date', 'Genotype', 'FEED1']
#     for col in essential_cols:
#         if col not in df_light.columns:
#             st.error(f"Light Phase Analysis: Essential column '{col}' is missing. Cannot proceed.")
#             return None

#     current_vars_to_analyze = [var for var in vars_to_analyze if var in df_light.columns]
#     if not current_vars_to_analyze:
#         st.error("Light Phase Analysis: None of the specified variables for analysis are present in the data.")
#         return None
        
#     try:
#         daily_feed_sum_light = df_light.groupby(['CHAN', 'Date', 'Genotype'])['FEED1'].sum().reset_index(name='Daily_Total_FEED1')
#         daily_feed_sum_light = daily_feed_sum_light.sort_values(['CHAN', 'Date'])
#     except KeyError as e:
#         st.error(f"Light Phase Analysis: Error grouping for daily feed. Missing column: {e}")
#         return None

#     if 'CHAN' in daily_feed_sum_light.columns and 'Daily_Total_FEED1' in daily_feed_sum_light.columns:
#         daily_feed_sum_light['Cumulative_Daily_FEED1'] = daily_feed_sum_light.groupby('CHAN')['Daily_Total_FEED1'].cumsum()
#     else:
#         daily_feed_sum_light['Cumulative_Daily_FEED1'] = np.nan

#     avg_daily_feed_light = daily_feed_sum_light.groupby(['CHAN', 'Genotype'])['Daily_Total_FEED1'].mean().reset_index(name='Avg_Daily_FEED1')

#     import numpy as np
#     import pandas as pd
#     summary_list_light = []
#     df_light_wt_ko = df_light[df_light['Genotype'].isin(['WT', 'KO'])]
#     print("df_light_wt_ko preview:")
#     print(df_light_wt_ko.head())

#     if df_light_wt_ko.empty:
#         st.warning("Light Phase Analysis: No data found for WT or KO genotypes.")

#     for var in current_vars_to_analyze:
#         print(f"\n--- Analyzing variable: '{var}' ---")
        
#         if var not in df_light_wt_ko.columns:
#             print(f"⚠️ Skipping '{var}' — not found in dataframe columns.")
#             continue

#         for genotype_to_check in ['WT', 'KO']: 
#             print(f"Processing genotype: {genotype_to_check}")
            
#             if genotype_to_check in df_light_wt_ko['Genotype'].unique():
#                 raw_data = df_light_wt_ko[df_light_wt_ko['Genotype'] == genotype_to_check][var]
#                 print(f"Raw data (first 5 rows) for {var} - {genotype_to_check}:\n{raw_data.head().tolist()}")

#                 data = pd.to_numeric(raw_data, errors='coerce').dropna()
#                 print(f"Cleaned numeric data (first 5 rows) for {var} - {genotype_to_check}:\n{data.head().tolist()}")
#                 print(f"Number of valid numeric entries: {len(data)}")

#                 if not data.empty:
#                     mean_val = data.mean()
#                     sem_val = data.sem()
#                     print(f"✅ Mean: {mean_val:.4f}, SEM: {sem_val:.4f}")
#                     summary_list_light.append({
#                         'variable': var,
#                         'Genotype': genotype_to_check,
#                         'mean': mean_val,
#                         'sem': sem_val
#                     })
#                 else:
#                     print(f"⚠️ No valid numeric data for {var} - {genotype_to_check}.")
#                     summary_list_light.append({
#                         'variable': var,
#                         'Genotype': genotype_to_check,
#                         'mean': np.nan,
#                         'sem': np.nan
#                     })
#             else:
#                 print(f"⚠️ Genotype '{genotype_to_check}' not found in data.")
#                 summary_list_light.append({
#                     'variable': var,
#                     'Genotype': genotype_to_check,
#                     'mean': np.nan,
#                     'sem': np.nan
#                 })

#     print("\n✅ Summary List:")
#     for entry in summary_list_light:
#         print(entry)

#     summary_df_light = pd.DataFrame(summary_list_light)
#     print("\n✅ Summary DataFrame Preview:")
#     print(summary_df_light.head())

#     ttest_results_light = []

#     print("🔍 Starting T-test analysis for light phase...")

#     # Check for WT and KO in df_light_wt_ko
#     if not df_light_wt_ko.empty and set(['WT', 'KO']).issubset(df_light_wt_ko['Genotype'].unique()):
#         print("✅ Light phase data contains both WT and KO genotypes.")
        
#         for var in current_vars_to_analyze:
#             print(f"\nAnalyzing variable: {var}")

#             data_wt = pd.to_numeric(df_light_wt_ko[df_light_wt_ko['Genotype'] == 'WT'][var], errors='coerce').dropna()
#             data_ko = pd.to_numeric(df_light_wt_ko[df_light_wt_ko['Genotype'] == 'KO'][var], errors='coerce').dropna()
            
#             print(f"WT sample size: {len(data_wt)}, KO sample size: {len(data_ko)}")

#             if len(data_wt) >= 2 and len(data_ko) >= 2:
#                 try:
#                     t_stat, p_val = ttest_ind(data_wt, data_ko, equal_var=False, nan_policy='omit')
#                     d = cohen_d(data_wt, data_ko)
#                     star = significance_stars(p_val)
#                     print(f"T-test success: t={t_stat:.3f}, p={p_val:.4g}, Cohen's d={d:.3f}, significance={star}")
#                     ttest_results_light.append({
#                         'variable': var,
#                         't_stat': t_stat,
#                         'p_value': p_val,
#                         'significance': star,
#                         'cohen_d': d
#                     })
#                 except Exception as e:
#                     st.warning(f"T-test failed for variable {var} in light phase: {e}")
#                     print(f"⚠️ T-test failed for {var}: {e}")
#                     ttest_results_light.append({'variable': var, 't_stat': np.nan, 'p_value': np.nan, 'significance': 'error', "cohen_d": np.nan})
#             else:
#                 print(f"⚠️ Not enough data for variable {var} (WT: {len(data_wt)}, KO: {len(data_ko)}). Skipping.")
#     else:
#         print("⚠️ Light phase data is missing WT or KO genotype.")

#     # T-test on average daily feed
#     if not avg_daily_feed_light.empty and 'Genotype' in avg_daily_feed_light.columns and 'Avg_Daily_FEED1' in avg_daily_feed_light.columns:
#         avg_daily_feed_light_wt_ko = avg_daily_feed_light[avg_daily_feed_light['Genotype'].isin(['WT', 'KO'])]
#         if set(['WT', 'KO']).issubset(avg_daily_feed_light_wt_ko['Genotype'].unique()):
#             avg_feed_wt = pd.to_numeric(avg_daily_feed_light_wt_ko[avg_daily_feed_light_wt_ko['Genotype'] == 'WT']['Avg_Daily_FEED1'], errors='coerce').dropna()
#             avg_feed_ko = pd.to_numeric(avg_daily_feed_light_wt_ko[avg_daily_feed_light_wt_ko['Genotype'] == 'KO']['Avg_Daily_FEED1'], errors='coerce').dropna()

#             print(f"\nAnalyzing Avg_Daily_FEED1 - WT: {len(avg_feed_wt)}, KO: {len(avg_feed_ko)}")

#             if len(avg_feed_wt) >= 2 and len(avg_feed_ko) >= 2:
#                 try:
#                     t_stat_feed, p_val_feed = ttest_ind(avg_feed_wt, avg_feed_ko, equal_var=False, nan_policy='omit')
#                     d_feed = cohen_d(avg_feed_wt, avg_feed_ko)
#                     star_feed = significance_stars(p_val_feed)
#                     print(f"Avg_Daily_FEED1 t={t_stat_feed:.3f}, p={p_val_feed:.4g}, d={d_feed:.3f}, sig={star_feed}")
#                     ttest_results_light.append({
#                         'variable': 'Avg_Daily_FEED1',
#                         't_stat': t_stat_feed,
#                         'p_value': p_val_feed,
#                         'significance': star_feed,
#                         'cohen_d': d_feed
#                     })
#                 except Exception as e:
#                     st.warning(f"T-test failed for Avg_Daily_FEED1 in light phase: {e}")
#                     print(f"⚠️ T-test failed for Avg_Daily_FEED1: {e}")
#                     ttest_results_light.append({'variable': 'Avg_Daily_FEED1', 't_stat': np.nan, 'p_value': np.nan, 'significance': 'error', "cohen_d": np.nan})
#             else:
#                 print("⚠️ Not enough data for Avg_Daily_FEED1. Skipping.")
#     else:
#         print("⚠️ avg_daily_feed_light is missing required columns or is empty.")

#     # Convert results to DataFrame
#     ttest_df_light = pd.DataFrame(ttest_results_light)
#     print("✅ T-test analysis for light phase complete.")

#     # Outlier Detection
#     if not avg_daily_feed_light.empty and 'Avg_Daily_FEED1' in avg_daily_feed_light.columns:
#         Q1 = avg_daily_feed_light['Avg_Daily_FEED1'].quantile(0.25)
#         Q3 = avg_daily_feed_light['Avg_Daily_FEED1'].quantile(0.75)
#         IQR = Q3 - Q1
#         if pd.isna(IQR) or IQR == 0:
#             avg_daily_feed_light['Outlier'] = False
#             print("⚠️ IQR is 0 or NaN, marking all as non-outliers.")
#         else:
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
#             avg_daily_feed_light['Outlier'] = (avg_daily_feed_light['Avg_Daily_FEED1'] < lower_bound) | (avg_daily_feed_light['Avg_Daily_FEED1'] > upper_bound)
#             outlier_count = avg_daily_feed_light['Outlier'].sum()
#             print(f"✅ Outlier detection complete. {outlier_count} outliers found.")
#     elif not avg_daily_feed_light.empty:
#         avg_daily_feed_light['Outlier'] = False
#         print("⚠️ Avg_Daily_FEED1 column not found, marking all as non-outliers.")

#     # Return the analysis results
#     return {
#         'summary_df': summary_df_light,
#         'ttest_df': ttest_df_light,
#         'avg_daily_feed': avg_daily_feed_light,
#         'cumulative_feed_data_for_plot': daily_feed_sum_light
#     }
def perform_light_phase_analysis_updated(df_light_input, vars_to_analyze=['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT']):
    import numpy as np
    from scipy.stats import ttest_ind
    
    if df_light_input is None or df_light_input.empty:
        return None

    df_light = df_light_input.copy()

    essential_cols = ['CHAN', 'Date', 'Genotype', 'FEED1']
    for col in essential_cols:
        if col not in df_light.columns:
            print(f"Essential column '{col}' is missing.")
            return None

    # Convert vars_to_analyze columns to numeric (coerce errors)
    for var in vars_to_analyze:
        if var in df_light.columns:
            df_light[var] = pd.to_numeric(df_light[var], errors='coerce')

    # Also convert FEED1 to numeric
    if 'FEED1' in df_light.columns:
        df_light['FEED1'] = pd.to_numeric(df_light['FEED1'], errors='coerce')

    current_vars_to_analyze = [var for var in vars_to_analyze if var in df_light.columns]
    if not current_vars_to_analyze:
        print("None of the specified variables for analysis are present in the data.")
        return None

    daily_avg = df_light.groupby(['CHAN', 'Date', 'Genotype'])[current_vars_to_analyze].mean().reset_index()

    daily_feed_sum_light = df_light.groupby(['CHAN', 'Date', 'Genotype'])['FEED1'].sum().reset_index(name='Daily_Total_FEED1')
    daily_feed_sum_light = daily_feed_sum_light.sort_values(['CHAN', 'Date'])
    daily_feed_sum_light['Cumulative_Daily_FEED1'] = daily_feed_sum_light.groupby('CHAN')['Daily_Total_FEED1'].cumsum()

    daily_avg = pd.merge(daily_avg, daily_feed_sum_light, on=['CHAN', 'Date', 'Genotype'], how='left')

    avg_daily_feed_light = daily_feed_sum_light.groupby(['CHAN', 'Genotype'])['Daily_Total_FEED1'].mean().reset_index(name='Avg_Daily_FEED1')

    summary_list_light = []
    df_light_wt_ko = daily_avg[daily_avg['Genotype'].isin(['WT', 'KO'])]

    for var in current_vars_to_analyze:
        for genotype_to_check in ['WT', 'KO']:
            data = df_light_wt_ko[df_light_wt_ko['Genotype'] == genotype_to_check][var]
            data = pd.to_numeric(data, errors='coerce').dropna()
            summary_list_light.append({
                'variable': var,
                'Genotype': genotype_to_check,
                'mean': data.mean() if not data.empty else np.nan,
                'sem': data.sem() if not data.empty else np.nan
            })

    summary_df_light = pd.DataFrame(summary_list_light)

    ttest_results_light = []
    for var in current_vars_to_analyze:
        data_wt = df_light_wt_ko[df_light_wt_ko['Genotype'] == 'WT'][var].dropna()
        data_ko = df_light_wt_ko[df_light_wt_ko['Genotype'] == 'KO'][var].dropna()
        if len(data_wt) >= 2 and len(data_ko) >= 2:
            t_stat, p_val = ttest_ind(data_wt, data_ko, equal_var=False, nan_policy='omit')
            d = cohen_d(data_wt, data_ko)
            star = significance_stars(p_val)
            ttest_results_light.append({
                'variable': var,
                't_stat': t_stat,
                'p_value': p_val,
                'significance': star,
                'cohen_d': d
            })

    avg_daily_feed_light_wt_ko = avg_daily_feed_light[avg_daily_feed_light['Genotype'].isin(['WT', 'KO'])]
    if set(['WT', 'KO']).issubset(avg_daily_feed_light_wt_ko['Genotype'].unique()):
        avg_feed_wt = avg_daily_feed_light_wt_ko[avg_daily_feed_light_wt_ko['Genotype'] == 'WT']['Avg_Daily_FEED1']
        avg_feed_ko = avg_daily_feed_light_wt_ko[avg_daily_feed_light_wt_ko['Genotype'] == 'KO']['Avg_Daily_FEED1']
        if len(avg_feed_wt) >= 2 and len(avg_feed_ko) >= 2:
            t_stat_feed, p_val_feed = ttest_ind(avg_feed_wt, avg_feed_ko, equal_var=False, nan_policy='omit')
            d_feed = cohen_d(avg_feed_wt, avg_feed_ko)
            star_feed = significance_stars(p_val_feed)
            ttest_results_light.append({
                'variable': 'Avg_Daily_FEED1',
                't_stat': t_stat_feed,
                'p_value': p_val_feed,
                'significance': star_feed,
                'cohen_d': d_feed
            })

    ttest_df_light = pd.DataFrame(ttest_results_light)

    if not avg_daily_feed_light.empty and 'Avg_Daily_FEED1' in avg_daily_feed_light.columns:
        Q1 = avg_daily_feed_light['Avg_Daily_FEED1'].quantile(0.25)
        Q3 = avg_daily_feed_light['Avg_Daily_FEED1'].quantile(0.75)
        IQR = Q3 - Q1
        if pd.isna(IQR) or IQR == 0:
            avg_daily_feed_light['Outlier'] = False
        else:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            avg_daily_feed_light['Outlier'] = (avg_daily_feed_light['Avg_Daily_FEED1'] < lower_bound) | (avg_daily_feed_light['Avg_Daily_FEED1'] > upper_bound)

    return {
        'summary_df': summary_df_light,
        'ttest_df': ttest_df_light,
        'avg_daily_feed': avg_daily_feed_light,
        'cumulative_feed_data_for_plot': daily_feed_sum_light
    }

def plot_light_phase_analysis(df_light_input_for_plot, analysis_results_for_plot, vars_to_analyze_for_plot=['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT']):
    plots = {}
    if not analysis_results_for_plot: return plots
    if df_light_input_for_plot is None or df_light_input_for_plot.empty: return plots

    summary_df = analysis_results_for_plot.get('summary_df')
    ttest_df = analysis_results_for_plot.get('ttest_df')
    avg_daily_feed = analysis_results_for_plot.get('avg_daily_feed')
    cumulative_feed_data_for_plot = analysis_results_for_plot.get('cumulative_feed_data_for_plot')

    actual_vars_in_summary = []
    if summary_df is not None and 'variable' in summary_df.columns:
        for var_check in vars_to_analyze_for_plot:
            if var_check in summary_df['variable'].unique():
                 if not summary_df[(summary_df['variable']==var_check) & (summary_df['mean'].notna())].empty:
                      actual_vars_in_summary.append(var_check)

    if 'Genotype' not in df_light_input_for_plot.columns: return plots
    df_light_wt_ko_plot = df_light_input_for_plot[df_light_input_for_plot['Genotype'].isin(['WT', 'KO'])]
    if df_light_wt_ko_plot.empty: return plots
    
    # Change here to fix order: WT on left, KO on right
    genotypes_sorted = ['WT', 'KO']  
    x_pos = np.arange(len(genotypes_sorted))

    fig1_light, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    plot_count = 0
    for i, var in enumerate(actual_vars_in_summary):
        ax = axes[i]
        plot_count += 1
        if summary_df is None or not all(col in summary_df.columns for col in ['variable', 'Genotype', 'mean', 'sem']):
            ax.text(0.5, 0.5, f"Summary data for {var} error.", ha='center', va='center', transform=ax.transAxes)
            continue

        data_plot = summary_df[(summary_df['variable'] == var) & (summary_df['Genotype'].isin(genotypes_sorted))].set_index('Genotype').reindex(genotypes_sorted)
        
        if not data_plot.empty and not data_plot['mean'].isnull().all():
            means = pd.to_numeric(data_plot['mean'].values, errors='coerce')
            sems = pd.to_numeric(data_plot['sem'].values, errors='coerce')
            valid_plot_indices = ~np.isnan(means)

            if not np.any(valid_plot_indices):
                 ax.text(0.5, 0.5, f"Numeric data for {var} error.", ha='center', va='center', transform=ax.transAxes)
                 continue
            
            bar_colors = ['#66c2a5' if g == 'WT' else '#fc8d62' for g in np.array(genotypes_sorted)[valid_plot_indices]]
            try:
                ax.bar(x_pos[valid_plot_indices], means[valid_plot_indices], yerr=sems[valid_plot_indices] if np.any(valid_plot_indices) else None, capsize=5, color=bar_colors, ecolor='dimgray', error_kw={'capthick':1.5, 'elinewidth':1.5})
                ax.set_xticks(x_pos)
                ax.set_xticklabels(genotypes_sorted)
                ax.set_title(f'{var} (Mean ± SEM) - Light Phase')
                ax.set_ylabel(var)
                
                if ttest_df is not None and not ttest_df.empty and 'variable' in ttest_df.columns and var in ttest_df['variable'].values and len(genotypes_sorted) == 2:
                    pval_row = ttest_df[ttest_df['variable'] == var]
                    if not pval_row.empty:
                        star = pval_row['significance'].iloc[0]
                        if star != 'ns':
                            current_means = means[valid_plot_indices]
                            current_sems = sems[valid_plot_indices] if np.any(valid_plot_indices) else np.zeros_like(current_means)
                            current_sems = np.nan_to_num(current_sems)

                            if len(current_means) > 0:
                                y_top = (current_means + current_sems).max()
                                y_min_val = current_means.min()
                                line_y = y_top * 1.05 if y_top >= 0 else y_top * 0.95
                                if y_top == y_min_val : line_y = y_top + 0.05 * abs(y_top) if y_top !=0 else 0.1
                                ax.plot(x_pos, [line_y, line_y], color='black', lw=1)
                                ax.text(np.mean(x_pos), line_y, star, ha='center', va='bottom', fontsize=14)
            except Exception as e:
                ax.text(0.5, 0.5, f"Plotting error for {var}: {e}", ha='center', va='center', transform=ax.transAxes, wrap=True)
                st.warning(f"Could not generate bar plot for {var} in light phase: {e}")
        else:
            ax.text(0.5, 0.5, f"Data for {var} is empty/NaN.", ha='center', va='center', transform=ax.transAxes)

    # ... rest of your function unchanged ...

    feed_plot_ax_idx = plot_count
    if feed_plot_ax_idx < len(axes):
        ax = axes[feed_plot_ax_idx]
        plot_count +=1
        if avg_daily_feed is not None and not avg_daily_feed.empty and \
           'Genotype' in avg_daily_feed.columns and 'Avg_Daily_FEED1' in avg_daily_feed.columns:
            
            feed_summary_plot = avg_daily_feed[avg_daily_feed['Genotype'].isin(genotypes_sorted)].groupby('Genotype')['Avg_Daily_FEED1'].agg(['mean', 'sem']).reindex(genotypes_sorted)
            
            if not feed_summary_plot.empty and 'mean' in feed_summary_plot.columns and 'sem' in feed_summary_plot.columns and \
               not feed_summary_plot['mean'].isnull().all():
                
                means = pd.to_numeric(feed_summary_plot['mean'].values, errors='coerce')
                sems = pd.to_numeric(feed_summary_plot['sem'].values, errors='coerce')
                valid_plot_indices = ~np.isnan(means)

                if not np.any(valid_plot_indices):
                    ax.text(0.5, 0.5, "Numeric Avg Daily FEED1 error.", ha='center', va='center', transform=ax.transAxes)
                else:
                    try:
                        bar_colors = ['#66c2a5' if g == 'WT' else '#fc8d62' for g in np.array(genotypes_sorted)[valid_plot_indices]]
                        ax.bar(x_pos[valid_plot_indices], means[valid_plot_indices], yerr=sems[valid_plot_indices] if np.any(valid_plot_indices) else None, capsize=5, color=bar_colors, ecolor='dimgray', error_kw={'capthick':1.5, 'elinewidth':1.5})
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(genotypes_sorted)
                        ax.set_title('Avg Daily FEED1 (Mean ± SEM) - Light Phase')
                        ax.set_ylabel('g (average daily intake)')

                        if ttest_df is not None and not ttest_df.empty and 'variable' in ttest_df.columns and \
                           'Avg_Daily_FEED1' in ttest_df['variable'].values and len(genotypes_sorted) == 2:
                            pval_row = ttest_df[ttest_df['variable'] == 'Avg_Daily_FEED1']
                            if not pval_row.empty:
                                star = pval_row['significance'].iloc[0]
                                if star != 'ns':
                                    current_means = means[valid_plot_indices]
                                    current_sems = sems[valid_plot_indices] if np.any(valid_plot_indices) else np.zeros_like(current_means)
                                    current_sems = np.nan_to_num(current_sems)
                                    if len(current_means) > 0:
                                        y_top = (current_means + current_sems).max()
                                        y_min_val = current_means.min()
                                        line_y = y_top * 1.05 if y_top >= 0 else y_top * 0.95
                                        if y_top == y_min_val : line_y = y_top + 0.05 * abs(y_top) if y_top !=0 else 0.1
                                        ax.plot(x_pos, [line_y, line_y], color='black', lw=1)
                                        ax.text(np.mean(x_pos), line_y, star, ha='center', va='bottom', fontsize=14)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Plotting error for Avg FEED1: {e}", ha='center', va='center', transform=ax.transAxes, wrap=True)
                        st.warning(f"Could not generate bar plot for Avg_Daily_FEED1 in light phase: {e}")
            else:
                ax.text(0.5, 0.5, "Avg Daily FEED1 summary error.", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Avg daily feed raw data error.", ha='center', va='center', transform=ax.transAxes)

    for i in range(plot_count, len(axes)): # Remove unused subplots
        if i < len(axes) and fig1_light.axes[i] is not None:
             try:
                 fig1_light.delaxes(axes[i])
             except Exception as e:
                 st.warning(f"Minor issue cleaning up light phase plot axes: {e}")


    fig1_light.suptitle('Mean ± SEM by Genotype during Light Phase', fontsize=17, y=1.00)
    fig1_light.tight_layout(rect=[0, 0.03, 1, 0.96])
    plots['mean_sem_plot'] = fig1_light

    fig2_light, ax2 = plt.subplots(figsize=(13, 7))
    try:
        if cumulative_feed_data_for_plot is not None and not cumulative_feed_data_for_plot.empty and \
           all(col in cumulative_feed_data_for_plot.columns for col in ['Genotype', 'Date', 'Cumulative_Daily_FEED1']) and \
           not cumulative_feed_data_for_plot['Cumulative_Daily_FEED1'].isnull().all() :
            
            df_for_cum_plot = cumulative_feed_data_for_plot.copy()
            df_for_cum_plot['Date'] = pd.to_datetime(df_for_cum_plot['Date'], errors='coerce')
            df_for_cum_plot.dropna(subset=['Date', 'Cumulative_Daily_FEED1'], inplace=True)
            
            df_for_cum_plot_wt_ko = df_for_cum_plot[df_for_cum_plot['Genotype'].isin(['WT', 'KO'])]

            if not df_for_cum_plot_wt_ko.empty:
                for genotype, color in zip(['WT', 'KO'], ['#66c2a5', '#fc8d62']):
                    if genotype in df_for_cum_plot_wt_ko['Genotype'].unique():
                        plot_data = df_for_cum_plot_wt_ko[df_for_cum_plot_wt_ko['Genotype'] == genotype]
                        plot_data = plot_data.sort_values(by='Date')
                        
                        mean_cum = plot_data.groupby('Date')['Cumulative_Daily_FEED1'].agg(['mean', 'sem']).reset_index()

                        if not mean_cum.empty and 'mean' in mean_cum.columns and 'sem' in mean_cum.columns and not mean_cum['mean'].isnull().all():
                            mean_val = pd.to_numeric(mean_cum['mean'], errors='coerce')
                            sem_val = pd.to_numeric(mean_cum['sem'], errors='coerce')

                            valid_indices_cum = ~mean_val.isnull()
                            if not np.any(valid_indices_cum): continue

                            ax2.plot(mean_cum['Date'][valid_indices_cum], mean_val[valid_indices_cum], label=genotype, color=color, marker='o', linestyle='-', markersize=5, linewidth=1.5)
                            sem_val_filled = sem_val[valid_indices_cum].fillna(0)
                            ax2.fill_between(mean_cum['Date'][valid_indices_cum],
                                             mean_val[valid_indices_cum] - sem_val_filled,
                                             mean_val[valid_indices_cum] + sem_val_filled,
                                             color=color, alpha=0.15)
                ax2.set_xlabel('Date', fontsize=12)
                ax2.set_ylabel('Mean Cumulative Daily Food Intake (g)', fontsize=12)
                ax2.set_title('Mean Cumulative Daily Food Intake Over Light Days', fontsize=14)
                if ax2.has_data():
                     ax2.legend(title='Genotype', fontsize=10)
                ax2.tick_params(axis='x', rotation=30, labelsize=10)
                ax2.tick_params(axis='y', labelsize=10)
                ax2.grid(True, linestyle='--', alpha=0.5)
            else:
                ax2.text(0.5, 0.5, "WT/KO cumulative feed data error for light phase plot.", ha='center', va='center', transform=ax2.transAxes, wrap=True)
        else:
            ax2.text(0.5, 0.5, "Cumulative feed raw data error for light phase plot.", ha='center', va='center', transform=ax2.transAxes, wrap=True)
    except Exception as e:
        ax2.text(0.5, 0.5, f"Cumulative plot error (Light): {e}", ha='center', va='center', transform=ax2.transAxes, wrap=True)
        st.warning(f"Could not generate cumulative feed plot for light phase: {e}")

    fig2_light.tight_layout()
    plots['cumulative_feed_plot'] = fig2_light
    
    return plots

# def perform_anova_analysis(df_light_input, df_dark_input, vars_to_analyze=['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT', 'FEED1']):
#     print("Starting ANOVA analysis...")
#     if df_light_input is None or df_light_input.empty or df_dark_input is None or df_dark_input.empty:
#         print("Warning: Light or Dark phase data is missing or empty.")
#         st.warning("ANOVA: Light or Dark phase data is missing or empty.")
#         return None

#     df_light = df_light_input.copy()
#     df_dark = df_dark_input.copy()
#     print(f"Initial Light phase data shape: {df_light.shape}")
#     print(f"Initial Dark phase data shape: {df_dark.shape}")

#     # Ensure numeric columns
#     for df_phase in [df_light, df_dark]:
#         if 'FEED1' in df_phase.columns:
#             df_phase['FEED1'] = pd.to_numeric(df_phase['FEED1'], errors='coerce')
#         for var_an in vars_to_analyze:
#             if var_an in df_phase.columns:
#                 df_phase[var_an] = pd.to_numeric(df_phase[var_an], errors='coerce')

#     df_dark['Phase'] = 'Dark'
#     df_light['Phase'] = 'Light'

#     df_combined = pd.concat([df_dark, df_light], ignore_index=True)
#     print(f"Combined data shape before genotype filtering: {df_combined.shape}")

#     df_combined = df_combined[df_combined['Genotype'].isin(['WT', 'KO'])]
#     print(f"Combined data shape after filtering for WT/KO genotypes: {df_combined.shape}")

#     if df_combined.empty:
#         print("Warning: Combined data is empty after filtering for WT/KO genotypes.")
#         st.warning("ANOVA: Combined data is empty after filtering for WT/KO genotypes.")
#         return None

#     df_combined['Genotype'] = pd.Categorical(df_combined['Genotype'], categories=['WT', 'KO'], ordered=True)
#     df_combined['Phase'] = pd.Categorical(df_combined['Phase'], categories=['Light', 'Dark'], ordered=True)

#     anova_results = []

#     for var in vars_to_analyze:
#         print(f"\nAnalyzing variable: {var}")
#         if var not in df_combined.columns:
#             print(f"Warning: Variable '{var}' not found in combined data. Skipping.")
#             st.warning(f"ANOVA: Variable '{var}' not found in combined data. Skipping.")
#             continue

#         if df_combined[var].isnull().all():
#             print(f"Warning: Variable '{var}' is all NaN. Skipping.")
#             st.warning(f"ANOVA: Variable '{var}' is all NaN. Skipping.")
#             continue

#         group_counts = df_combined.dropna(subset=[var]).groupby(['Genotype', 'Phase']).size()
#         print(f"Group counts for '{var}':\n{group_counts}")

#         if len(group_counts) < 4 or (group_counts < 2).any():
#             print(f"Warning: Insufficient data for variable '{var}' across Genotype/Phase combinations. Skipping.")
#             st.warning(f"ANOVA: Insufficient data for variable '{var}' across Genotype/Phase combinations. Skipping.\nGroup counts:\n{group_counts}")
#             continue

#         try:
#             # Removed backticks here to fix formula parsing error
#             formula = f"{var} ~ C(Genotype) + C(Phase) + C(Genotype):C(Phase)"
#             model = ols(formula, data=df_combined).fit()
#             anova_table = sm.stats.anova_lm(model, typ=2)
#             print(f"ANOVA table for {var}:\n{anova_table}")

#             res_dict = {'Variable': var}
#             if 'C(Genotype)' in anova_table.index:
#                 res_dict['F_Genotype'] = anova_table.loc['C(Genotype)', 'F']
#                 res_dict['p_Genotype'] = anova_table.loc['C(Genotype)', 'PR(>F)']
#             if 'C(Phase)' in anova_table.index:
#                 res_dict['F_Phase'] = anova_table.loc['C(Phase)', 'F']
#                 res_dict['p_Phase'] = anova_table.loc['C(Phase)', 'PR(>F)']
#             if 'C(Genotype):C(Phase)' in anova_table.index:
#                 res_dict['F_Interaction'] = anova_table.loc['C(Genotype):C(Phase)', 'F']
#                 res_dict['p_Interaction'] = anova_table.loc['C(Genotype):C(Phase)', 'PR(>F)']

#             anova_results.append(res_dict)
#         except Exception as e:
#             print(f"Error during ANOVA for variable {var}: {e}")
#             st.error(f"ANOVA error for variable {var}: {e}")
#             anova_results.append({
#                 'Variable': var, 'F_Genotype': np.nan, 'p_Genotype': np.nan,
#                 'F_Phase': np.nan, 'p_Phase': np.nan,
#                 'F_Interaction': np.nan, 'p_Interaction': np.nan,
#                 'Error': str(e)
#             })

#     if not anova_results:
#         print("Warning: No ANOVA results generated. Check input data and variables.")
#         st.warning("ANOVA: No results were generated. Check data and variable selection.")
#         return None

#     anova_df = pd.DataFrame(anova_results)

#     # Add significance stars
#     if 'p_Genotype' in anova_df.columns:
#         anova_df['sig_Genotype'] = anova_df['p_Genotype'].apply(significance_stars)
#     if 'p_Phase' in anova_df.columns:
#         anova_df['sig_Phase'] = anova_df['p_Phase'].apply(significance_stars)
#     if 'p_Interaction' in anova_df.columns:
#         anova_df['sig_Interaction'] = anova_df['p_Interaction'].apply(significance_stars)

#     print("ANOVA analysis completed successfully.")
#     return anova_df
def perform_anova_analysis(df_light_input, df_dark_input, vars_to_analyze=['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT', 'FEED1']):
    import pandas as pd
    import numpy as np
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    from scipy.stats import f_oneway
    import streamlit as st  # assuming st is used in your environment

    print("Starting ANOVA analysis...")
    
    if df_light_input is None or df_light_input.empty or df_dark_input is None or df_dark_input.empty:
        print("Warning: Light or Dark phase data is missing or empty.")
        st.warning("ANOVA: Light or Dark phase data is missing or empty.")
        return None

    # Convert columns to numeric to avoid aggregation errors
    for df_phase in [df_light_input, df_dark_input]:
        if 'FEED1' in df_phase.columns:
            df_phase['FEED1'] = pd.to_numeric(df_phase['FEED1'], errors='coerce')
        for var in vars_to_analyze:
            if var in df_phase.columns:
                df_phase[var] = pd.to_numeric(df_phase[var], errors='coerce')

    group_cols = ['CHAN', 'Date', 'Genotype']
    
    # Aggregate to daily means
    df_light = df_light_input.groupby(group_cols)[vars_to_analyze].mean().reset_index()
    df_light['Phase'] = 'Light'
    
    df_dark = df_dark_input.groupby(group_cols)[vars_to_analyze].mean().reset_index()
    df_dark['Phase'] = 'Dark'

    # Combine data
    df_combined = pd.concat([df_light, df_dark], ignore_index=True)
    print(f"Combined data shape before genotype filtering: {df_combined.shape}")

    # Filter to WT and KO genotypes
    df_combined = df_combined[df_combined['Genotype'].isin(['WT', 'KO'])]
    print(f"Combined data shape after filtering for WT/KO genotypes: {df_combined.shape}")

    if df_combined.empty:
        print("Warning: Combined data is empty after filtering for WT/KO genotypes.")
        st.warning("ANOVA: Combined data is empty after filtering for WT/KO genotypes.")
        return None

    # Set Genotype and Phase as categorical for modeling
    df_combined['Genotype'] = pd.Categorical(df_combined['Genotype'], categories=['WT', 'KO'], ordered=True)
    df_combined['Phase'] = pd.Categorical(df_combined['Phase'], categories=['Light', 'Dark'], ordered=True)

    anova_results = []

    for var in vars_to_analyze:
        print(f"\nAnalyzing variable: {var}")
        if var not in df_combined.columns:
            print(f"Warning: Variable '{var}' not found in combined data. Skipping.")
            st.warning(f"ANOVA: Variable '{var}' not found in combined data. Skipping.")
            continue

        if df_combined[var].isnull().all():
            print(f"Warning: Variable '{var}' is all NaN. Skipping.")
            st.warning(f"ANOVA: Variable '{var}' is all NaN. Skipping.")
            continue

        group_counts = df_combined.dropna(subset=[var]).groupby(['Genotype', 'Phase']).size()
        print(f"Group counts for '{var}':\n{group_counts}")

        if len(group_counts) < 4 or (group_counts < 2).any():
            print(f"Warning: Insufficient data for variable '{var}' across Genotype/Phase combinations. Skipping.")
            st.warning(f"ANOVA: Insufficient data for variable '{var}' across Genotype/Phase combinations. Skipping.\nGroup counts:\n{group_counts}")
            continue

        try:
            formula = f"{var} ~ C(Genotype) + C(Phase) + C(Genotype):C(Phase)"
            model = ols(formula, data=df_combined).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(f"ANOVA table for {var}:\n{anova_table}")

            res_dict = {'Variable': var}
            if 'C(Genotype)' in anova_table.index:
                res_dict['F_Genotype'] = anova_table.loc['C(Genotype)', 'F']
                res_dict['p_Genotype'] = anova_table.loc['C(Genotype)', 'PR(>F)']
            if 'C(Phase)' in anova_table.index:
                res_dict['F_Phase'] = anova_table.loc['C(Phase)', 'F']
                res_dict['p_Phase'] = anova_table.loc['C(Phase)', 'PR(>F)']
            if 'C(Genotype):C(Phase)' in anova_table.index:
                res_dict['F_Interaction'] = anova_table.loc['C(Genotype):C(Phase)', 'F']
                res_dict['p_Interaction'] = anova_table.loc['C(Genotype):C(Phase)', 'PR(>F)']

            anova_results.append(res_dict)

        except Exception as e:
            print(f"Error during ANOVA for variable {var}: {e}")
            st.error(f"ANOVA error for variable {var}: {e}")
            anova_results.append({
                'Variable': var, 'F_Genotype': np.nan, 'p_Genotype': np.nan,
                'F_Phase': np.nan, 'p_Phase': np.nan,
                'F_Interaction': np.nan, 'p_Interaction': np.nan,
                'Error': str(e)
            })

    if not anova_results:
        print("Warning: No ANOVA results generated. Check input data and variables.")
        st.warning("ANOVA: No results were generated. Check data and variable selection.")
        return None

    anova_df = pd.DataFrame(anova_results)

    # Add significance stars
    if 'p_Genotype' in anova_df.columns:
        anova_df['sig_Genotype'] = anova_df['p_Genotype'].apply(significance_stars)
    if 'p_Phase' in anova_df.columns:
        anova_df['sig_Phase'] = anova_df['p_Phase'].apply(significance_stars)
    if 'p_Interaction' in anova_df.columns:
        anova_df['sig_Interaction'] = anova_df['p_Interaction'].apply(significance_stars)

    print("ANOVA analysis completed successfully.")
    return anova_df


from statannotations.Annotator import Annotator

def plot_anova_interaction_plots(df_light_input, df_dark_input, vars_to_analyze=['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT', 'FEED1']):
    plot_figures_list = []
    if df_light_input is None or df_light_input.empty or df_dark_input is None or df_dark_input.empty:
        return {'interaction_plots_list': plot_figures_list}  # Return empty list in dict

    df_light = df_light_input.copy()
    df_dark = df_dark_input.copy()

    # Ensure 'FEED1' and other vars are numeric
    for df_phase in [df_light, df_dark]:
        if 'FEED1' in df_phase.columns:
            df_phase['FEED1'] = pd.to_numeric(df_phase['FEED1'], errors='coerce')
        for var_an_plot in vars_to_analyze:
            if var_an_plot in df_phase.columns:
                df_phase[var_an_plot] = pd.to_numeric(df_phase[var_an_plot], errors='coerce')

    df_dark['Phase'] = 'Dark'
    df_light['Phase'] = 'Light'
    df_combined = pd.concat([df_dark, df_light], ignore_index=True)
    df_combined = df_combined[df_combined['Genotype'].isin(['WT', 'KO'])]

    if df_combined.empty:
        return {'interaction_plots_list': plot_figures_list}

    df_combined['Genotype'] = pd.Categorical(df_combined['Genotype'], categories=['WT', 'KO'], ordered=True)
    df_combined['Phase'] = pd.Categorical(df_combined['Phase'], categories=['Light', 'Dark'], ordered=True)

    for var in vars_to_analyze:
        if var not in df_combined.columns or df_combined[var].isnull().all():
            st.write(f"Skipping interaction plot for {var}: not in data or all NaN.")
            continue

        summary_check = df_combined.dropna(subset=[var]).groupby(['Genotype', 'Phase'], observed=False)[var].agg(['mean', 'sem']).reset_index()
        if summary_check.empty or summary_check['mean'].isnull().all() or summary_check['sem'].isnull().all():
            st.write(f"Skipping interaction plot for {var}: insufficient data for mean/sem calculation.")
            continue
        if len(summary_check['Phase'].unique()) < 2 or len(summary_check['Genotype'].unique()) < 2:
            st.write(f"Skipping interaction plot for {var}: requires at least two levels for both Genotype and Phase.")
            continue

        fig, ax = plt.subplots(figsize=(6, 5))

        try:
            sns.barplot(
                x='Genotype',
                y='mean',
                hue='Phase',
                data=summary_check,
                ax=ax,
                errorbar=None,
                palette={'Light': '#FFD700', 'Dark': '#4B0082'}
            )

            # Sort summary_check to match bar order
            summary_check_sorted = summary_check.sort_values(['Genotype', 'Phase']).reset_index(drop=True)

            # Add error bars by matching each bar to the row in summary_check_sorted
            for idx, row in summary_check_sorted.iterrows():
                if idx >= len(ax.patches):
                    break
                bar = ax.patches[idx]
                sem_value = row['sem']
                if pd.notna(sem_value):
                    x = bar.get_x() + bar.get_width() / 2
                    y = bar.get_height()
                    ax.errorbar(x, y, yerr=sem_value, fmt='none', c='black', capsize=3, elinewidth=1)

            # Prepare raw data for stats annotation
            df_for_stats = df_combined[['Genotype', 'Phase', var]].dropna()

            pairs = [
                (("WT", "Light"), ("KO", "Light")),
                (("WT", "Dark"), ("KO", "Dark"))
            ]

            annotator = Annotator(
                ax,
                pairs=pairs,
                data=df_for_stats,
                x='Genotype',
                y=var,
                hue='Phase',
                order=['WT', 'KO'],
                hue_order=['Light', 'Dark']
            )

            annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=0)
            annotator.apply_and_annotate()

            ax.set_title(f'{var} (Mean ± SEM)', fontsize=12)
            ax.set_ylabel(var)
            ax.set_xlabel("Genotype")
            ax.legend(title='Phase')
            plt.tight_layout()

            plot_figures_list.append(fig)

        except Exception as e:
            st.error(f"Error plotting interaction for {var}: {e}")
            plt.close(fig)

    return {'interaction_plots_list': plot_figures_list}



st.title('Animal Data Analysis App')

st.sidebar.header('Upload and Settings')
uploaded_file = st.sidebar.file_uploader('Upload your ZIP folder containing Oxymax CSV files:', type=['zip'])
days_to_exclude = st.sidebar.number_input('Days to exclude from the beginning:', min_value=0, value=1, step=1)
feed1_lower_bound = st.sidebar.number_input('FEED1 lower bound (values below this will be set to 0):', value=0.0)
feed1_upper_bound = st.sidebar.number_input('FEED1 upper bound (values above this will be set to 0):', value=0.5)

def process_zip(uploaded_file_obj):
    if uploaded_file_obj is None:
        st.info("Please upload a ZIP file.")
        return None

    extract_path = "./oxymax_data"
    os.makedirs(extract_path, exist_ok=True)

    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file_obj.getvalue())) as z:
            z.extractall(extract_path)
        st.success(f"Extracted files to: {extract_path}")
    except zipfile.BadZipFile:
        st.error("Error: Uploaded file is not a valid ZIP file or is corrupted.")
        return None
    except Exception as e:
        st.error(f"An error occurred during ZIP extraction: {e}")
        return None

    csv_files = []
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        st.warning("No CSV files found in the uploaded ZIP archive.")
        return []
    else:
        st.info(f"Found {len(csv_files)} CSV files.")
        return csv_files

def process_csv_files(csv_files, days_to_exclude, feed1_lower_bound, feed1_upper_bound):
    metadata_rows = []
    dataframes = []

    if not csv_files:
        st.warning("No CSV files provided to process.")
        return None, None, None, None, None

    for file in csv_files:
        try:
            basename = os.path.basename(file)
            meta_dict = {'Filename': basename}

            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline().strip() for _ in range(22)]

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'Subject ID':
                        meta_dict['Subject ID'] = value
                    elif key == 'Body Mass (g)':
                        meta_dict['Body Mass (g)'] = value
                    elif key == 'Cage ID':
                        meta_dict['CageID'] = value # Ensure this matches later use
                    elif key == 'Data Filename':
                        meta_dict['Data Filename'] = value
            
            if "KO" in basename.upper(): # Case-insensitive check
                meta_dict['Genotype'] = "KO"
            elif "WT" in basename.upper(): # Case-insensitive check
                meta_dict['Genotype'] = "WT"
            else:
                meta_dict['Genotype'] = "Unknown"
            
            metadata_rows.append(meta_dict)

            df = pd.read_csv(file, skiprows=22)
            df.columns = [col.strip() for col in df.columns]
            
            # Clean rows: remove ':EVENTS', '#', 'NaN', or empty strings in the first column
            if df.empty:
                st.write(f"Skipping empty file (after header): {basename}")
                continue

            first_col = df.columns[0]
            df = df[~df[first_col].astype(str).str.contains(':EVENTS|#|NaN|^$', na=True)] # Handle actual NaN and empty strings

            df = df[df['DATE/TIME'].astype(str).str.contains('/', na=False)]
            df['DATE/TIME'] = pd.to_datetime(df['DATE/TIME'], errors='coerce')
            df.dropna(subset=['DATE/TIME'], inplace=True)
            # Remove the filter on '/' because your datetime format uses '-'
            # # Parse datetime with explicit format to avoid warnings
            # df['DATE/TIME'] = pd.to_datetime(df['DATE/TIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

            # # Drop rows where datetime parsing failed
            # df.dropna(subset=['DATE/TIME'], inplace=True)

            if 'LED' in df.columns:
                df['LED'] = pd.to_numeric(df['LED'], errors='coerce')
                df['Day/Night'] = np.where(df['LED'] > 50, 'Day', 'Night')
            else:
                df['Day/Night'] = 'Unknown'

            df['Genotype'] = meta_dict.get('Genotype', 'Unknown')
            df['CageID'] = meta_dict.get('CageID', 'Unknown')
            # 'Filename' (basename) is already in meta_dict, adding it to df for consistency
            df['Filename'] = basename 
            
            dataframes.append(df)
            st.write(f"Processed {basename} — {len(df)} rows")

        except Exception as e:
            st.error(f"Error processing file {file}: {e}")
            continue # Skip to next file if error

    if not dataframes:
        st.warning("No dataframes were created. Check CSV files and processing logic.")
        return None, None, None, None, None

    combined_df = pd.concat(dataframes, ignore_index=True)
    metadata_df = pd.DataFrame(metadata_rows)

    # Re-parse 'DATE/TIME' in combined_df (already done per file, but good for consistency)
    combined_df['DATE/TIME'] = pd.to_datetime(combined_df['DATE/TIME'], errors='coerce')
    combined_df.dropna(subset=['DATE/TIME'], inplace=True)
    
    if combined_df.empty:
        st.warning("Combined dataframe is empty after processing all files.")
        return None, metadata_df, None, None, None # Return metadata_df as it might have info

    # Filter by Date Range
    min_dt = combined_df['DATE/TIME'].min()
    cutoff_time = min_dt + pd.Timedelta(days=days_to_exclude)
    df_filtered = combined_df[combined_df['DATE/TIME'] >= cutoff_time].copy() # Use .copy()

    if df_filtered.empty:
        st.warning(f"Dataframe is empty after excluding the first {days_to_exclude} days.")
        return combined_df, metadata_df, None, None, None


    # Clean FEED1
    if 'FEED1' in df_filtered.columns:
        df_filtered['FEED1'] = pd.to_numeric(df_filtered['FEED1'], errors='coerce').fillna(0)
        df_filtered.loc[df_filtered['FEED1'] < feed1_lower_bound, 'FEED1'] = 0
        df_filtered.loc[df_filtered['FEED1'] > feed1_upper_bound, 'FEED1'] = 0
    else:
        st.warning("'FEED1' column not found. Skipping FEED1 cleaning.")
        # Add an empty FEED1 column if it doesn't exist to prevent errors later? Or handle downstream.
        # For now, let it proceed, downstream steps might fail or need adjustment.

    # Classify Light/Dark Phases based on 'LED SATURATION'
    if 'LED SATURATION' in df_filtered.columns:
        df_filtered['LED SATURATION'] = pd.to_numeric(df_filtered['LED SATURATION'], errors='coerce') # Ensure numeric
        phase_map = {0: 'Light', 100: 'Dark'}
        df_filtered['Phase'] = df_filtered['LED SATURATION'].map(phase_map)
        # Filter to keep only rows where 'Phase' is 'Light' or 'Dark'
        df_filtered = df_filtered[df_filtered['Phase'].isin(['Light', 'Dark'])].copy() # Use .copy()
        if df_filtered.empty:
            st.warning("Dataframe is empty after 'Phase' (Light/Dark) classification based on 'LED SATURATION'. Check 'LED SATURATION' values (should be 0 or 100).")
            # Return what we have so far, as light/dark splits will be empty
            return combined_df, metadata_df, None, None, df_filtered 
    else:
        st.warning("'LED SATURATION' column not found. Cannot classify Light/Dark phases. Subsequent analysis might be affected.")
        # If 'LED SATURATION' is critical, we might return earlier or handle differently
        # For now, df_light and df_dark will be empty if this column is missing.
        df_light = pd.DataFrame() 
        df_dark = pd.DataFrame()
        # Return all available dataframes
        return combined_df, metadata_df, df_light, df_dark, df_filtered


    # Split into Light and Dark DataFrames
    df_light = df_filtered[df_filtered['Phase'] == 'Light'].copy()
    df_dark = df_filtered[df_filtered['Phase'] == 'Dark'].copy()

    # Calculate Cumulative FEED1 for df_light
    if not df_light.empty and 'FEED1' in df_light.columns and 'CHAN' in df_light.columns:
        df_light = df_light.sort_values(by=['CHAN', 'DATE/TIME'])
        df_light['Date'] = df_light['DATE/TIME'].dt.date
        df_light['Cumulative FEED1'] = df_light.groupby(['CHAN', 'Date'])['FEED1'].cumsum()
    elif 'FEED1' not in df_light.columns:
        st.warning("Cannot calculate 'Cumulative FEED1' for light phase: 'FEED1' column missing.")
    elif 'CHAN' not in df_light.columns:
        st.warning("Cannot calculate 'Cumulative FEED1' for light phase: 'CHAN' column missing.")


    # Calculate Cumulative FEED1 for df_dark
    if not df_dark.empty and 'FEED1' in df_dark.columns and 'CHAN' in df_dark.columns:
        df_dark = df_dark.sort_values(by=['CHAN', 'DATE/TIME'])
        df_dark['Date'] = df_dark['DATE/TIME'].dt.date
        df_dark['Cumulative FEED1'] = df_dark.groupby(['CHAN', 'Date'])['FEED1'].cumsum()
    elif 'FEED1' not in df_dark.columns:
        st.warning("Cannot calculate 'Cumulative FEED1' for dark phase: 'FEED1' column missing.")
    elif 'CHAN' not in df_dark.columns:
        st.warning("Cannot calculate 'Cumulative FEED1' for dark phase: 'CHAN' column missing.")

    return combined_df, metadata_df, df_light, df_dark, df_filtered

# Main application flow
if uploaded_file is not None:
    # Check if the uploaded file is new or if data hasn't been processed yet
    # For simplicity, we'll re-process if a file is present.
    # A more robust check might involve file ID or a manual reset button.
    
    csv_files = process_zip(uploaded_file)

    if csv_files:
        processed_data = process_csv_files(csv_files, days_to_exclude, feed1_lower_bound, feed1_upper_bound)
        
        if processed_data and processed_data[0] is not None: # Check if combined_df is not None
            combined_df, metadata_df, df_light, df_dark, df_filtered = processed_data
            
            # Store data in session state
            st.session_state['combined_df'] = combined_df
            st.session_state['metadata_df'] = metadata_df
            st.session_state['df_light'] = df_light
            st.session_state['df_dark'] = df_dark
            st.session_state['df_filtered'] = df_filtered
            st.session_state['data_processed'] = True
            
            # Display initial summaries
            if st.session_state.get('data_processed'): # Check if data is processed
                metadata_df_to_display = st.session_state.get('metadata_df')
                if metadata_df_to_display is not None: 
                    st.header("Subject Metadata Summary")
                    st.dataframe(metadata_df_to_display)
                    # Download button for metadata_df - Placed here to be associated with its display
                    if not metadata_df_to_display.empty:
                         st.download_button(
                            label="Download Metadata Summary (CSV)",
                            data=metadata_df_to_display.to_csv(index=False).encode('utf-8'),
                            file_name='metadata_summary.csv',
                            mime='text/csv',
                            key='download_metadata_csv_key_v4' # Ensure key is unique if regenerating
                        )
                
                current_combined_df = st.session_state.get('combined_df')
                # Logic for current_combined_df display and download
                if current_combined_df is not None and not current_combined_df.empty:
                    st.header("Preview of Cleaned and Filtered Data (from combined_df)")
                    st.dataframe(current_combined_df.head())
                    
                    if 'DATE/TIME' in current_combined_df.columns:
                        min_dt = current_combined_df['DATE/TIME'].min()
                        max_dt = current_combined_df['DATE/TIME'].max()
                        if pd.notna(min_dt) and pd.notna(max_dt):
                             st.write(f"📅 Data range in combined_df: {min_dt.strftime('%Y-%m-%d %H:%M')} to {max_dt.strftime('%Y-%m-%d %H:%M')}")
                        else:
                            st.warning("Could not determine data range from combined_df (min/max dates are NaT).")
                    
                        # Download button for combined_df
                        st.download_button(
                            label="Download Combined Cleaned Data (CSV)",
                            data=current_combined_df.to_csv(index=False).encode('utf-8'),
                            file_name='combined_cleaned_data.csv',
                            mime='text/csv',
                            key='download_combined_csv'
                        )
                    # THIS ELSE IS FOR: if 'DATE/TIME' in current_combined_df.columns:
                    else: 
                        st.warning("'DATE/TIME' column not found in combined_df, cannot display data range.")
                # This ELIF corresponds to: if current_combined_df is not None and not current_combined_df.empty:
                elif current_combined_df is not None and current_combined_df.empty: 
                    st.warning("The combined_df is empty after processing. Cannot display preview or data range.")
                # Implicit else for current_combined_df being None is acceptable as earlier processing would have handled it or shown errors.

                # Download buttons for metadata and light/dark phase data are now placed after the combined_df display logic,
                # ensuring the if/elif chain for current_combined_df is contiguous.
                # These will show if data_processed is true and their respective data is available.
                
                if st.session_state.get('metadata_df') is not None and not st.session_state.get('metadata_df').empty:
                    st.download_button(
                        label="Download Metadata Summary (CSV)",
                        data=st.session_state['metadata_df'].to_csv(index=False).encode('utf-8'),
                        file_name='metadata_summary.csv',
                        mime='text/csv',
                        key='download_metadata_csv'
                    )

                df_light_for_excel = st.session_state.get('df_light')
                df_dark_for_excel = st.session_state.get('df_dark')
                if df_light_for_excel is not None and df_dark_for_excel is not None:
                    excel_data_map = {}
                    if not df_light_for_excel.empty:
                        excel_data_map['Light_Phase_Data'] = df_light_for_excel
                    if not df_dark_for_excel.empty:
                        excel_data_map['Dark_Phase_Data'] = df_dark_for_excel
                    
                    if excel_data_map: 
                        excel_bytes = dfs_to_excel_bytes(excel_data_map)
                        st.download_button(
                            label="Download Light & Dark Phase Data (Excel)",
                            data=excel_bytes,
                            file_name='light_dark_phase_data.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key='download_light_dark_excel'
                        )


        else: # This ELSE is for: if processed_data and processed_data[0] is not None:
            st.error("Failed to process CSV files. Please check the files and settings.")
            st.session_state['data_processed'] = False # Reset flag if processing fails
    else:
        st.warning("No CSV files were found in the uploaded ZIP. Processing cannot continue.")
        st.session_state['data_processed'] = False # Reset flag

# In the main app flow, after initial data display section
if st.session_state.get('data_processed') and 'df_dark' in st.session_state:
    df_dark_data_main = st.session_state.get('df_dark') 
    if isinstance(df_dark_data_main, pd.DataFrame) and not df_dark_data_main.empty:
        st.markdown("---") 
        st.header("🌙 Dark Phase Analysis")
        
        vars_to_analyze_dark_options = ['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT'] 

        run_analysis_button_dark = st.button("Run Dark Phase Analysis", key="run_dark_analysis_button_key")

        if run_analysis_button_dark:
            with st.spinner("Performing dark phase analysis..."):
                analysis_results_dark = perform_dark_phase_analysis(df_dark_data_main, vars_to_analyze_dark_options) 
                st.session_state['dark_analysis_results'] = analysis_results_dark
                if analysis_results_dark: 
                    phase_plots_dark = plot_dark_phase_analysis(df_dark_data_main, analysis_results_dark, vars_to_analyze_dark_options)
                    st.session_state['dark_phase_plots'] = phase_plots_dark
                    if analysis_results_dark.get('summary_df') is not None and not analysis_results_dark.get('summary_df').empty:
                        st.session_state['dark_analysis_done'] = True 
                        st.success("Dark phase analysis complete.")
                    else:
                        st.session_state['dark_analysis_done'] = False
                        st.warning("Dark phase analysis completed, but no summary data was generated.")
                        st.session_state['dark_phase_plots'] = {} 
                else: 
                    st.error("Dark phase analysis failed or returned no results.")
                    st.session_state['dark_analysis_done'] = False
                    st.session_state['dark_analysis_results'] = None 
                    st.session_state['dark_phase_plots'] = None 
        
    if st.session_state.get('dark_analysis_done'):
        retrieved_dark_results_display = st.session_state.get('dark_analysis_results')
        retrieved_dark_plots_display = st.session_state.get('dark_phase_plots')

        if retrieved_dark_results_display:
            st.subheader("Summary Statistics (Dark Phase)")
            summary_table_display = retrieved_dark_results_display.get('summary_df')
            if summary_table_display is not None and not summary_table_display.empty:
                # Format only numeric columns safely
                format_dict = {col: "{:.3g}" for col in summary_table_display.select_dtypes(include='number').columns}
                st.dataframe(summary_table_display.style.format(na_rep='-', formatter=format_dict))
            else:
                st.write("No summary statistics for dark phase.")
            
            st.subheader("T-test Results (WT vs KO - Dark Phase)")
            ttest_table_display = retrieved_dark_results_display.get('ttest_df')
            if ttest_table_display is not None and not ttest_table_display.empty:
                # Only format columns present and numeric
                format_dict = {
                    't_stat': "{:.2f}",
                    'p_value': "{:.3e}",
                    'cohen_d': "{:.2f}"
                }
                format_dict = {k: v for k, v in format_dict.items() if k in ttest_table_display.columns and pd.api.types.is_numeric_dtype(ttest_table_display[k])}
                st.dataframe(ttest_table_display.style.format(na_rep='-', formatter=format_dict))
            else:
                st.write("No T-test results for dark phase.")

            st.subheader("Outliers in Average Daily FEED1 (Dark Phase)")
            outliers_table_display = retrieved_dark_results_display.get('avg_daily_feed')
            if (outliers_table_display is not None and not outliers_table_display.empty and
                    'Outlier' in outliers_table_display.columns):
                st.dataframe(outliers_table_display[outliers_table_display['Outlier'] == True].style.format(na_rep='-'))
            else:
                st.write("No outlier data for average daily FEED1 in dark phase.")

        # Download button for Dark Phase Analysis Tables
        if retrieved_dark_results_display:
            dark_excel_map = {}
            summary_df_dark = retrieved_dark_results_display.get('summary_df')
            ttest_df_dark = retrieved_dark_results_display.get('ttest_df')
            avg_daily_feed_dark = retrieved_dark_results_display.get('avg_daily_feed')
            
            if summary_df_dark is not None and not summary_df_dark.empty:
                dark_excel_map['Summary_Stats_Dark'] = summary_df_dark
            if ttest_df_dark is not None and not ttest_df_dark.empty:
                dark_excel_map['Ttest_Results_Dark'] = ttest_df_dark
            if avg_daily_feed_dark is not None and not avg_daily_feed_dark.empty:
                dark_excel_map['Avg_Daily_Feed_Outliers_Dark'] = avg_daily_feed_dark
            
            if dark_excel_map:
                dark_analysis_excel_bytes = dfs_to_excel_bytes(dark_excel_map)
                st.download_button(
                    label="Download Dark Phase Analysis Tables (Excel)",
                    data=dark_analysis_excel_bytes,
                    file_name='dark_phase_analysis_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='download_dark_analysis_excel'
                )

        if retrieved_dark_plots_display:
            st.subheader("Mean ± SEM by Genotype (Dark Phase)")
            mean_sem_p_display = retrieved_dark_plots_display.get('mean_sem_plot')
            if mean_sem_p_display and hasattr(mean_sem_p_display, 'number_of_axes') and mean_sem_p_display.number_of_axes > 0: 
                st.pyplot(mean_sem_p_display)
            else:
                st.write("Mean ± SEM plot for dark phase not generated.")
            
            st.subheader("Mean Cumulative Daily Food Intake (Dark Phase)")
            cum_feed_p_display = retrieved_dark_plots_display.get('cumulative_feed_plot')
            if cum_feed_p_display and hasattr(cum_feed_p_display, 'number_of_axes') and cum_feed_p_display.number_of_axes > 0: 
                st.pyplot(cum_feed_p_display)
            else:
                st.write("Cumulative food intake plot for dark phase not generated.")

            # Download button for Dark Phase Plots
            plot_files_dark_map = {}
            mean_sem_plot_dark_fig_for_download = retrieved_dark_plots_display.get('mean_sem_plot')
            cumulative_feed_plot_dark_fig_for_download = retrieved_dark_plots_display.get('cumulative_feed_plot')
            
            if mean_sem_plot_dark_fig_for_download: 
                plot_files_dark_map['dark_mean_sem_plot.png'] = fig_to_bytes(mean_sem_plot_dark_fig_for_download)
            if cumulative_feed_plot_dark_fig_for_download: 
                plot_files_dark_map['dark_cumulative_feed_plot.png'] = fig_to_bytes(cumulative_feed_plot_dark_fig_for_download)
            
            if plot_files_dark_map:
                zip_bytes_dark = create_zip_from_plot_bytes(plot_files_dark_map)
                if zip_bytes_dark:
                    st.download_button(
                        label="Download Dark Phase Plots (ZIP)", 
                        data=zip_bytes_dark, 
                        file_name='dark_phase_plots.zip', 
                        mime='application/zip', 
                        key='download_dark_plots_zip'
                    )
                
    elif run_analysis_button_dark : 
        st.warning("Dark phase analysis was attempted but did not complete successfully.")

    elif st.session_state.get('data_processed'): 
        st.info("Dark phase data is not available/empty; analysis section skipped.")

# Light Phase Analysis Section
if st.session_state.get('data_processed') and 'df_light' in st.session_state:
    df_light_data_main = st.session_state.get('df_light')
    if isinstance(df_light_data_main, pd.DataFrame) and not df_light_data_main.empty:
        st.markdown("---")
        st.header("☀️ Light Phase Analysis")

        vars_to_analyze_light_options = ['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT']

        run_analysis_button_light = st.button("Run Light Phase Analysis", key="run_light_analysis_button_key")

        if run_analysis_button_light:
            with st.spinner("Performing light phase analysis..."):
                analysis_results_light = perform_light_phase_analysis(df_light_data_main, vars_to_analyze_light_options)
                st.session_state['light_analysis_results'] = analysis_results_light
                if analysis_results_light:
                    phase_plots_light = plot_light_phase_analysis(df_light_data_main, analysis_results_light, vars_to_analyze_light_options)
                    st.session_state['light_phase_plots'] = phase_plots_light
                    if analysis_results_light.get('summary_df') is not None and not analysis_results_light.get('summary_df').empty:
                        st.session_state['light_analysis_done'] = True
                        st.success("Light phase analysis complete.")
                    else:
                        st.session_state['light_analysis_done'] = False
                        st.warning("Light phase analysis completed, but no summary data was generated.")
                        st.session_state['light_phase_plots'] = {}
                else:
                    st.error("Light phase analysis failed or returned no results.")
                    st.session_state['light_analysis_done'] = False
                    st.session_state['light_analysis_results'] = None
                    st.session_state['light_phase_plots'] = None

        if st.session_state.get('light_analysis_done'):
            retrieved_light_results_display = st.session_state.get('light_analysis_results')
            retrieved_light_plots_display = st.session_state.get('light_phase_plots')

            if retrieved_light_results_display:
                st.subheader("Summary Statistics (Light Phase)")
                summary_table_light_display = retrieved_light_results_display.get('summary_df')
                if summary_table_light_display is not None and not summary_table_light_display.empty:
                    # Format numeric columns safely like dark phase
                    format_dict = {col: "{:.3g}" for col in summary_table_light_display.select_dtypes(include='number').columns}
                    st.dataframe(summary_table_light_display.style.format(na_rep='-', formatter=format_dict))
                else:
                    st.write("No summary statistics for light phase.")

                st.subheader("T-test Results (WT vs KO - Light Phase)")
                ttest_table_light_display = retrieved_light_results_display.get('ttest_df')
                if ttest_table_light_display is not None and not ttest_table_light_display.empty:
                    format_dict = {
                        't_stat': "{:.2f}",
                        'p_value': "{:.3e}",
                        'cohen_d': "{:.2f}"
                    }
                    format_dict = {k: v for k, v in format_dict.items() if k in ttest_table_light_display.columns and pd.api.types.is_numeric_dtype(ttest_table_light_display[k])}
                    st.dataframe(ttest_table_light_display.style.format(na_rep='-', formatter=format_dict))
                else:
                    st.write("No T-test results for light phase.")

                st.subheader("Outliers in Average Daily FEED1 (Light Phase)")
                outliers_table_light_display = retrieved_light_results_display.get('avg_daily_feed')
                if (outliers_table_light_display is not None and not outliers_table_light_display.empty and
                    'Outlier' in outliers_table_light_display.columns):
                    st.dataframe(outliers_table_light_display[outliers_table_light_display['Outlier'] == True].style.format(na_rep='-'))
                else:
                    st.write("No outlier data for average daily FEED1 in light phase.")

            # Download button for Light Phase Analysis Tables
            if retrieved_light_results_display:
                light_excel_map = {}
                summary_df_light = retrieved_light_results_display.get('summary_df')
                ttest_df_light = retrieved_light_results_display.get('ttest_df')
                avg_daily_feed_light = retrieved_light_results_display.get('avg_daily_feed')

                if summary_df_light is not None and not summary_df_light.empty:
                    light_excel_map['Summary_Stats_Light'] = summary_df_light
                if ttest_df_light is not None and not ttest_df_light.empty:
                    light_excel_map['Ttest_Results_Light'] = ttest_df_light
                if avg_daily_feed_light is not None and not avg_daily_feed_light.empty:
                    light_excel_map['Avg_Daily_Feed_Outliers_Light'] = avg_daily_feed_light

                if light_excel_map:
                    light_analysis_excel_bytes = dfs_to_excel_bytes(light_excel_map)
                    st.download_button(
                        label="Download Light Phase Analysis Tables (Excel)",
                        data=light_analysis_excel_bytes,
                        file_name='light_phase_analysis_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key='download_light_analysis_excel'
                    )

            if retrieved_light_plots_display:
                st.subheader("Mean ± SEM by Genotype (Light Phase)")
                mean_sem_p_light_display = retrieved_light_plots_display.get('mean_sem_plot')
                if mean_sem_p_light_display and hasattr(mean_sem_p_light_display, 'number_of_axes') and mean_sem_p_light_display.number_of_axes > 0:
                    st.pyplot(mean_sem_p_light_display)
                else:
                    st.write("Mean ± SEM plot for light phase not generated.")

                st.subheader("Mean Cumulative Daily Food Intake (Light Phase)")
                cum_feed_p_light_display = retrieved_light_plots_display.get('cumulative_feed_plot')
                if cum_feed_p_light_display and hasattr(cum_feed_p_light_display, 'number_of_axes') and cum_feed_p_light_display.number_of_axes > 0:
                    st.pyplot(cum_feed_p_light_display)
                else:
                    st.write("Cumulative food intake plot for light phase not generated.")

                # Download button for Light Phase Plots
                plot_files_light_map = {}
                mean_sem_plot_light_fig_for_download = retrieved_light_plots_display.get('mean_sem_plot')
                cumulative_feed_plot_light_fig_for_download = retrieved_light_plots_display.get('cumulative_feed_plot')

                if mean_sem_plot_light_fig_for_download:
                    plot_files_light_map['light_mean_sem_plot.png'] = fig_to_bytes(mean_sem_plot_light_fig_for_download)
                if cumulative_feed_plot_light_fig_for_download:
                    plot_files_light_map['light_cumulative_feed_plot.png'] = fig_to_bytes(cumulative_feed_plot_light_fig_for_download)

                if plot_files_light_map:
                    zip_bytes_light = create_zip_from_plot_bytes(plot_files_light_map)
                    if zip_bytes_light:
                        st.download_button(
                            label="Download Light Phase Plots (ZIP)",
                            data=zip_bytes_light,
                            file_name='light_phase_plots.zip',
                            mime='application/zip',
                            key='download_light_plots_zip'
                        )

        elif run_analysis_button_light:
            st.warning("Light phase analysis was attempted but did not complete successfully or produced no displayable output.")

    elif st.session_state.get('data_processed'):
        st.info("Light phase data is not available or is empty; analysis section skipped.")

# Combined ANOVA Analysis Section
import pandas as pd

def anova_results_have_errors(anova_df: pd.DataFrame) -> bool:
    """
    Check if ANOVA results DataFrame has 'Error' column and
    if all rows in 'Error' column indicate an error.
    Returns True if all are errors or dataframe is None/empty, else False.
    """
    if anova_df is None or anova_df.empty:
        return True  # No data means error condition

    if 'Error' not in anova_df.columns:
        # No 'Error' column means no errors reported
        return False

    # Define a function to interpret error values
    def is_error_val(val):
        if pd.isna(val):
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return len(val.strip()) > 0
        return False

    # Check if all rows indicate error
    all_errors = anova_df['Error'].apply(is_error_val).all()
    return all_errors

if st.session_state.get('data_processed'):
    df_light_data_for_anova = st.session_state.get('df_light')
    df_dark_data_for_anova = st.session_state.get('df_dark')

    if isinstance(df_light_data_for_anova, pd.DataFrame) and not df_light_data_for_anova.empty and \
       isinstance(df_dark_data_for_anova, pd.DataFrame) and not df_dark_data_for_anova.empty:

        st.markdown("---")
        st.header("📊 Combined Two-Way ANOVA (Genotype x Phase)")

        vars_for_anova = ['DO2', 'DCO2', 'RER', 'HEAT', 'XTOT', 'FEED1']

        run_anova_button = st.button("Run ANOVA Analysis", key="run_anova_button_key")

        if run_anova_button:
            with st.spinner("Performing ANOVA analysis and generating interaction plots..."):

                anova_results_df = perform_anova_analysis(df_light_data_for_anova, df_dark_data_for_anova, vars_for_anova)
                st.session_state['anova_results_df'] = anova_results_df

                if anova_results_df is not None and not anova_results_df.empty:
                    if not anova_results_have_errors(anova_results_df):
                        anova_plots_dict = plot_anova_interaction_plots(df_light_data_for_anova, df_dark_data_for_anova, vars_for_anova)
                        st.session_state['anova_interaction_plot_figures'] = anova_plots_dict.get('interaction_plots_list')
                        st.session_state['anova_analysis_done'] = True
                        st.success("ANOVA analysis complete.")
                    else:
                        st.error("ANOVA analysis completed, but all variables resulted in errors.")
                        st.session_state['anova_analysis_done'] = False
                        st.session_state['anova_interaction_plot_figures'] = None
                else:
                    st.error("ANOVA analysis failed or returned no results. Ensure sufficient data in both phases and WT/KO groups for the selected variables.")
                    st.session_state['anova_analysis_done'] = False
                    st.session_state['anova_results_df'] = None
                    st.session_state['anova_interaction_plot_figures'] = None

        if st.session_state.get('anova_analysis_done'):
            retrieved_anova_df = st.session_state.get('anova_results_df')
            retrieved_anova_plots_list = st.session_state.get('anova_interaction_plot_figures')

            if retrieved_anova_df is not None and not retrieved_anova_df.empty:
                st.subheader("ANOVA Results Table")
                cols_to_format_p = {col: "{:.3e}" for col in ['p_Genotype', 'p_Phase', 'p_Interaction'] if col in retrieved_anova_df.columns}
                cols_to_format_f = {col: "{:.2f}" for col in ['F_Genotype', 'F_Phase', 'F_Interaction'] if col in retrieved_anova_df.columns}
                cols_to_format_p.update(cols_to_format_f)

                styled_anova_df = retrieved_anova_df.style.format(cols_to_format_p, na_rep='-')
                st.dataframe(styled_anova_df)

                if retrieved_anova_df is not None and not retrieved_anova_df.empty:
                    anova_excel_bytes = dfs_to_excel_bytes({'ANOVA_Results': retrieved_anova_df})
                    st.download_button(
                        label="Download ANOVA Results Table (Excel)",
                        data=anova_excel_bytes,
                        file_name='anova_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key='download_anova_excel'
                    )

            if retrieved_anova_plots_list:
                st.subheader("Interaction Plots (Mean ± SEM)")
                for fig_anova in retrieved_anova_plots_list:
                    if fig_anova and hasattr(fig_anova, 'number_of_axes') and fig_anova.number_of_axes > 0:
                        st.pyplot(fig_anova)

                if retrieved_anova_plots_list and any(fig is not None for fig in retrieved_anova_plots_list):
                    plot_files_anova_map = {}
                    vars_for_anova_naming = st.session_state.get('vars_for_anova_plotting', vars_for_anova)
                    if 'anova_results_df' in st.session_state and st.session_state['anova_results_df'] is not None and 'Variable' in st.session_state['anova_results_df']:
                        vars_for_anova_naming = st.session_state['anova_results_df']['Variable'].tolist()

                    for i, fig_anova_dl in enumerate(retrieved_anova_plots_list):
                        if fig_anova_dl:
                            var_name = vars_for_anova_naming[i] if i < len(vars_for_anova_naming) else f"plot_{i+1}"
                            plot_bytes = fig_to_bytes(fig_anova_dl)
                            if plot_bytes:
                                plot_files_anova_map[f'anova_interaction_{var_name}.png'] = plot_bytes

                    if plot_files_anova_map:
                        zip_bytes_anova = create_zip_from_plot_bytes(plot_files_anova_map)
                        if zip_bytes_anova:
                            st.download_button(
                                label="Download ANOVA Interaction Plots (ZIP)",
                                data=zip_bytes_anova,
                                file_name='anova_interaction_plots.zip',
                                mime='application/zip',
                                key='download_anova_plots_zip'
                            )
                elif st.session_state.get('anova_analysis_done'):
                    st.write("Interaction plots could not be generated or are empty.")

        elif run_anova_button:
            st.warning("ANOVA analysis was attempted but did not complete successfully or produced no results. Please review data and messages above.")

    else:
        st.info("ANOVA analysis requires valid data from both Light and Dark phases. One or both are missing or empty.")
