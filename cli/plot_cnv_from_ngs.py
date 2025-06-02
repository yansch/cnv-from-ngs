#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from matplotlib import pyplot as plt
from scipy.stats import zscore

PLOT_Y_LIM = 10
GENE_ANNOTATION_PADDING = 0
ARM_COLORS = {'p': 'tab:blue', 'q': 'tab:orange', 'spanning': 'tab:green', None: 'grey'}

# Default paths (can be overridden by command-line arguments)
DEFAULT_CYTOBAND_PATH = Path('input/static/cytoBand.txt')
DEFAULT_RELEVANT_GENES_PATH = Path('input/static/relevant_genes.csv')
DEFAULT_OUTPUT_DIR = Path('output')

# Column definitions
CYTOBAND_COLUMN_NAMES = ['chromosome', 'start', 'end', 'band', 'giemsa']
RELEVANT_GENES_COLUMN_NAMES = ['gene', 'chromosome', 'start', 'end']


def _sanitize_chromosome_column(df, chromosome_col='chromosome'):
    """
    Converts a chromosome column to numeric Int64Dtype, removing 'chr' prefix.
    Rows with unparsable chromosome values (e.g., 'X', 'Y') are dropped.
    """
    if chromosome_col not in df.columns:
        print(f"Warning: Chromosome column '{chromosome_col}' not found.", file=sys.stderr)
        return df

    df = df.copy()
    # Ensure the column is treated as string initially for replacements
    df[chromosome_col] = df[chromosome_col].astype(str).str.replace('chr', '', regex=False)
    # Convert to numeric, coercing errors (like 'X', 'Y') to NaN
    df[chromosome_col] = pd.to_numeric(df[chromosome_col], errors='coerce')
    # Drop rows where conversion failed
    original_rows = len(df)
    df = df.dropna(subset=[chromosome_col])
    dropped_rows = original_rows - len(df)
    if dropped_rows > 0:
        print(f"Info: Dropped {dropped_rows} rows with non-numeric chromosome identifiers (e.g., X, Y).",
              file=sys.stderr)

    # Convert to nullable integer type
    df[chromosome_col] = df[chromosome_col].astype(pd.Int64Dtype())
    return df


def extract_case_identifier(filepath: Path) -> str:
    """Extracts a base identifier from the input filename."""
    # Example: Takes 'path/to/XYZ_123.cnr' -> 'XYZ_123'
    return filepath.stem.split('.')[0]


def load_and_preprocess_cytoband(file_path):
    """Loads cytoband data and preprocesses chromosome column."""
    try:
        df = pd.read_csv(file_path, delimiter='\t', names=CYTOBAND_COLUMN_NAMES)
    except FileNotFoundError:
        print(f"Error: Cytoband file not found at {file_path}", file=sys.stderr)
        sys.exit(1)  # Exit if essential static data is missing
    except Exception as e:
        print(f"Error loading cytoband file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    return _sanitize_chromosome_column(df, 'chromosome')


def calculate_chromosome_features(cytoband_df):
    """Calculates chromosome lengths and absolute start positions."""
    if cytoband_df.empty:
        print("Warning: Cytoband data is empty, cannot calculate chromosome features.", file=sys.stderr)
        return pd.DataFrame(columns=['chromosome', 'length', 'chromosome_absolute_start']), {}

    chromosomes_df = (
        cytoband_df
        .groupby('chromosome')
        .agg(length=('end', 'max'))
        .reset_index()
        .sort_values('chromosome')  # Ensure sorted order
    )
    # Calculate cumulative start positions
    chromosomes_df['chromosome_absolute_start'] = chromosomes_df['length'].cumsum() - chromosomes_df['length']
    # Create a mapping for easy lookup
    chromosome_start_map = chromosomes_df.set_index('chromosome')['chromosome_absolute_start'].to_dict()
    return chromosomes_df, chromosome_start_map


def create_arm_mapping_df(cytoband_df, chromosome_start_map):
    """Creates a DataFrame mapping chromosome arms to genomic and absolute coordinates."""
    if cytoband_df.empty or not chromosome_start_map:
        print("Warning: Cannot create arm mapping without cytoband data or chromosome map.", file=sys.stderr)
        return pd.DataFrame(columns=['chromosome', 'arm', 'arm_start', 'arm_end', 'arm_abs_start', 'arm_abs_end'])

    # Add 'arm' column based on the first character of the band name
    arm_df = cytoband_df.assign(arm=lambda x: x['band'].astype(str).str[0].str.lower())

    # Aggregate start/end for each arm
    arm_df = arm_df.groupby(['chromosome', 'arm']).agg(
        arm_start=('start', 'min'),
        arm_end=('end', 'max')
    ).reset_index()

    # Map chromosome absolute starts to the arm data
    chrom_starts_series = pd.Series(chromosome_start_map, name='chr_abs_start').reset_index()
    chrom_starts_series = chrom_starts_series.rename(columns={'index': 'chromosome'})  # Ensure column name matches

    # Merge and calculate absolute arm starts/ends
    arm_df = arm_df.merge(chrom_starts_series, on='chromosome', how='left')
    arm_df = arm_df.assign(
        arm_abs_start=lambda x: x['chr_abs_start'] + x['arm_start'],
        arm_abs_end=lambda x: x['chr_abs_start'] + x['arm_end']
    )

    # Select and order final columns
    return arm_df[['chromosome', 'arm', 'arm_start', 'arm_end', 'arm_abs_start', 'arm_abs_end']]


def create_arm_lookup_table(arm_mapping_df):
    """Pivots arm mapping data for easy lookup of p/q arm boundaries."""
    if arm_mapping_df.empty:
        print("Warning: Cannot create arm lookup table from empty arm mapping.", file=sys.stderr)
        return pd.DataFrame()

    try:
        # Pivot to get p/q start/end in columns
        arm_lookup = arm_mapping_df.pivot(
            index='chromosome',
            columns='arm',
            values=['arm_start', 'arm_end']
        )

        # Flatten MultiIndex columns (e.g., ('arm_start', 'p') -> 'arm_start_p')
        arm_lookup.columns = ['_'.join(col).strip() for col in arm_lookup.columns.values]

        # Rename columns to the desired format (e.g., 'p_start')
        rename_map = {
            'arm_start_p': 'p_start', 'arm_end_p': 'p_end',
            'arm_start_q': 'q_start', 'arm_end_q': 'q_end'
        }
        arm_lookup = arm_lookup.rename(columns=rename_map)

        # Ensure all four essential columns exist, adding NaN if missing
        for col in ['p_start', 'p_end', 'q_start', 'q_end']:
            if col not in arm_lookup.columns:
                arm_lookup[col] = np.nan

        return arm_lookup.reset_index()  # Keep chromosome as a column

    except Exception as e:
        print(f"Error creating arm lookup table: {e}", file=sys.stderr)
        # Return an empty df with expected columns on error
        return pd.DataFrame(columns=['chromosome', 'p_start', 'p_end', 'q_start', 'q_end'])


def load_and_preprocess_relevant_genes(file_path):
    """Loads relevant genes data and preprocesses chromosome column."""
    try:
        df = pd.read_csv(file_path, delimiter=';', names=RELEVANT_GENES_COLUMN_NAMES)
    except FileNotFoundError:
        print(f"Warning: Relevant genes file not found at {file_path}. Gene annotation will be skipped.",
              file=sys.stderr)
        return pd.DataFrame()  # Return empty DataFrame if genes are optional
    except Exception as e:
        print(f"Error loading relevant genes file {file_path}: {e}", file=sys.stderr)
        return pd.DataFrame()
    return _sanitize_chromosome_column(df, 'chromosome')


def build_gene_interval_trees(relevant_genes_df):
    """Builds a dictionary of IntervalTrees for gene lookups, per chromosome."""
    if relevant_genes_df.empty:
        return {}
    trees = {}
    # Ensure chromosome is integer type for dictionary keys
    relevant_genes_df = relevant_genes_df.dropna(subset=['chromosome'])
    relevant_genes_df['chromosome'] = relevant_genes_df['chromosome'].astype(int)

    for chrom, group in relevant_genes_df.groupby('chromosome'):
        tree = IntervalTree()
        for _, row in group.iterrows():
            # Define interval for gene based on its start and end from file
            # IntervalTree is exclusive for end, so use end + 1 if coordinates are inclusive
            start = int(row.start)
            end = int(row.end)  # Use gene end from file if available
            tree.addi(start, end + 1, {'gene': row.gene, 'start': start, 'end': end})
        trees[chrom] = tree
    return trees


def load_ngs_cnr_file(cnr_filepath, case_identifier):
    """Loads a single .cnr file, adds case identifier, and preprocesses."""
    if not cnr_filepath.exists():
        print(f"Error: Input .cnr file not found at {cnr_filepath}", file=sys.stderr)
        return None

    try:
        # Assuming standard .cnr format with headers
        df = pd.read_csv(cnr_filepath, sep='\t')
    except Exception as e:
        print(f"Error reading .cnr file {cnr_filepath}: {e}", file=sys.stderr)
        return None

    df['case_identifier'] = case_identifier  # Use the provided identifier

    # Drop original gene column if it exists, we re-annotate later
    df = df.drop(columns=['gene'], errors='ignore')

    # Sanitize chromosome column (handles 'chr' prefix and non-numeric values)
    df = _sanitize_chromosome_column(df)
    if df.empty:
        print(f"Warning: No data left in {cnr_filepath.name} after chromosome sanitation.", file=sys.stderr)
        return None

    # Ensure essential columns have correct types
    for col in ['start', 'end']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(pd.Int64Dtype())
    df = df.dropna(subset=['chromosome', 'start', 'end', 'log2'])  # Drop rows if key columns became NaN

    log2_series = df['log2']

    if len(log2_series.unique()) == 1:
        df.loc[:, 'log2'] = 0.0
    else:
        mean_val = log2_series.mean()
        std_val = log2_series.std()

        df.loc[:, 'log2'] = (log2_series - mean_val) / std_val

    return df


def add_segment_arm_classification(df, arm_lookup_table):
    """Classifies DataFrame segments/bins into chromosome arms (p, q, spanning)."""

    df['chromosome'] = df['chromosome'].astype(pd.Int64Dtype())
    for col in ['start', 'end']:
        if col not in df.columns:
            raise ValueError(f"Input DataFrame must contain '{col}' column for arm classification.")
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(pd.Int64Dtype())

    # Prepare arm_lookup_table and merge
    dfa = df  # Default to df if arm_lookup_table is unusable
    if not arm_lookup_table.empty and 'chromosome' in arm_lookup_table.columns:
        alt = arm_lookup_table.copy()
        alt['chromosome'] = pd.to_numeric(alt['chromosome'], errors='coerce')
        alt = alt.dropna(subset=['chromosome'])
        if not alt.empty:
            alt['chromosome'] = alt['chromosome'].astype(pd.Int64Dtype())
            dfa = df.merge(alt, on='chromosome', how='left')
        else:
            print("Warning: arm_lookup_table has no valid chromosome entries after cleaning. No arm data merged.")
    else:
        print("Warning: arm_lookup_table is empty or missing 'chromosome'. No arm data merged.")

    # Ensure arm boundary columns exist in dfa and are of correct type (Int64Dtype)
    arm_boundary_cols = ['p_start', 'p_end', 'q_start', 'q_end']
    for col in arm_boundary_cols:
        if col not in dfa.columns:  # If merge didn't happen or alt lacked these
            dfa[col] = pd.NA  # Add as all pd.NA
        dfa[col] = pd.to_numeric(dfa[col], errors='coerce').astype(pd.Int64Dtype())

    # Classify start/end positions of segments to p or q arms
    for pos_col in ['start', 'end']:
        p_start_cmp = dfa['p_start'].fillna(float('-inf'))
        p_end_cmp = dfa['p_end'].fillna(float('inf'))
        q_start_cmp = dfa['q_start'].fillna(float('-inf'))
        q_end_cmp = dfa['q_end'].fillna(float('inf'))

        is_in_p_arm = (dfa[pos_col] >= p_start_cmp) & (dfa[pos_col] < p_end_cmp)
        is_in_q_arm = (dfa[pos_col] >= q_start_cmp) & (dfa[pos_col] < q_end_cmp)

        # Convert to NumPy boolean arrays for np.select. pd.NA in conditions -> False.
        condlist = [
            is_in_p_arm.fillna(False).to_numpy(dtype=bool),
            is_in_q_arm.fillna(False).to_numpy(dtype=bool)
        ]
        choicelist = ['p', 'q']
        dfa[f'{pos_col}_arm'] = np.select(condlist, choicelist, default=None)

    # Determine final 'segment_arm' classification (p, q, spanning, or None)
    start_arm_series = dfa.get('start_arm', pd.Series(None, index=dfa.index, dtype=object))
    end_arm_series = dfa.get('end_arm', pd.Series(None, index=dfa.index, dtype=object))

    conditions_final = [
        (start_arm_series.notna()) & (start_arm_series == end_arm_series),  # Entirely within one arm
        (start_arm_series.notna()) & (end_arm_series.notna()) & (start_arm_series != end_arm_series)  # Spans p and q
    ]
    choices_final = [
        start_arm_series,  # Use the arm name (e.g., 'p' or 'q')
        'spanning'
    ]
    dfa['segment_arm'] = np.select(conditions_final, choices_final, default=None)

    # Clean up intermediate columns
    cols_to_drop = [c for c in arm_boundary_cols + ['start_arm', 'end_arm'] if c in dfa.columns]
    dfa = dfa.drop(columns=cols_to_drop)

    return dfa


def annotate_segments_with_genes(df, gene_interval_trees, padding=GENE_ANNOTATION_PADDING):
    """Annotates DataFrame segments with overlapping genes from interval trees."""
    if df.empty or not gene_interval_trees:
        df['gene'] = None
        return df

    gene_annotations = []
    for _, row in df.iterrows():
        chrom = int(row.chromosome)

        segment_start = int(row.start)
        segment_end = int(row.end)

        query_start = max(0, segment_start - padding)
        query_end = segment_end + 1  # query_end is exclusive for tree.overlap

        tree = gene_interval_trees.get(chrom)
        gene_found = None

        if tree:
            # Query interval tree with the padded window
            overlaps = tree.overlap(query_start, query_end)
            if overlaps:
                gene_found = sorted(overlaps)[0].data['gene']

        gene_annotations.append(gene_found)

    df['gene'] = gene_annotations
    return df


def prep_ngs_for_plotting(df, chrom_start_map):
    """Calculates absolute start position for plotting NGS bins."""
    if df.empty or not chrom_start_map:
        print("Warning: Cannot prepare NGS data for plotting without data or chromosome map.", file=sys.stderr)
        return df.assign(absolute_start=np.nan)  # Add column even if empty

    # Ensure chromosome type matches map keys (int)
    df['chromosome'] = df['chromosome'].astype(int)

    # Calculate absolute start based on bin midpoint for plotting scatter points
    # Check if map has necessary chromosome keys before mapping
    valid_chroms = df['chromosome'].isin(chrom_start_map.keys())
    if not valid_chroms.all():
        print(
            f"Warning: {sum(~valid_chroms)} bins have chromosomes not found in the chromosome map. Their absolute positions will be NaN.",
            file=sys.stderr)

    df['absolute_start'] = df['chromosome'].map(chrom_start_map) + ((df['start'] + df['end']) // 2)
    df['length'] = df['end'] - df['start']  # Needed for weighted aggregation
    return df


def aggregate_ngs_to_arms(classified_ngs_df, chrom_features_df):
    """Aggregates NGS log2 values to arm level using weighted average."""
    if (classified_ngs_df.empty or
            'segment_arm' not in classified_ngs_df.columns or
            chrom_features_df.empty or
            'length' not in classified_ngs_df.columns or
            'log2' not in classified_ngs_df.columns):
        print("Warning: Cannot aggregate NGS to arms due to missing data or columns (segment_arm, length, log2).",
              file=sys.stderr)
        return pd.DataFrame()

    # Filter for valid data: classified arm, positive length, non-NaN log2
    ngs_valid = classified_ngs_df[
        classified_ngs_df['segment_arm'].notna() &
        (classified_ngs_df['length'] > 0) &
        classified_ngs_df['log2'].notna()
        ].copy()

    if ngs_valid.empty:
        print("Warning: No valid NGS bins found for arm aggregation.", file=sys.stderr)
        return pd.DataFrame()

    # Ensure chromosome types are consistent (int)
    ngs_valid['chromosome'] = ngs_valid['chromosome'].astype(int)
    chrom_features_df['chromosome'] = chrom_features_df['chromosome'].astype(int)

    # Define weighted average function (handles potential empty groups)
    def weighted_avg(x, weights_col='length'):
        weights = ngs_valid.loc[x.index, weights_col]
        if x.empty or weights.empty or weights.sum() == 0:
            return np.nan
        return np.average(x, weights=weights)

    # Group by case, chromosome, and arm, then aggregate
    arm_agg = (
        ngs_valid.groupby(['case_identifier', 'chromosome', 'segment_arm'])
        .agg(
            log2=('log2', weighted_avg),
            arm_start=('start', 'min'),  # Min start of bins in the arm
            arm_end=('end', 'max'),  # Max end of bins in the arm
        ).reset_index()
    )

    # Add absolute start/end coordinates for the aggregated arm segment
    arm_agg = arm_agg.merge(
        chrom_features_df[['chromosome', 'chromosome_absolute_start']],
        on='chromosome',
        how='left'
    )
    arm_agg['arm_absolute_start'] = arm_agg['chromosome_absolute_start'] + arm_agg['arm_start']
    arm_agg['arm_absolute_end'] = arm_agg['chromosome_absolute_start'] + arm_agg['arm_end']

    # Clean up and return, dropping rows where log2 aggregation failed
    return arm_agg.drop(
        columns=['chromosome_absolute_start'], errors='ignore'
    ).dropna(subset=['log2'])


def aggregate_data_by_gene(annotated_df):
    """Aggregates log2 data to gene level based on annotation."""
    if (annotated_df.empty or
            'gene' not in annotated_df.columns or
            'absolute_start' not in annotated_df.columns or
            'log2' not in annotated_df.columns):
        print("Warning: Cannot aggregate by gene due to missing data or columns (gene, absolute_start, log2).",
              file=sys.stderr)
        return pd.DataFrame()

    # Filter for rows with valid gene annotations and data
    ngs_with_gene = annotated_df[annotated_df['gene'].notna()].copy()
    if ngs_with_gene.empty:
        return pd.DataFrame()

    # Group by gene and case, calculate median position and log2
    gene_reps = ngs_with_gene.groupby(['gene', 'case_identifier']).agg(
        absolute_position=('absolute_start', 'median'),  # Use median absolute start as representative position
        log2=('log2', 'median')  # Use median log2 of bins hitting the gene
    ).reset_index()

    # Clip log2 values to plot limits (redundant if already filtered, but safe)
    gene_reps['log2'] = gene_reps['log2'].clip(lower=-PLOT_Y_LIM, upper=PLOT_Y_LIM)

    return gene_reps


# --- Plotting Functions ---

def _plot_ngs_scatter(ax, ngs_df_case, arm_colors_dict):
    """Plots individual NGS bins as a scatter plot with alpha based on log2 z-score."""
    if ngs_df_case.empty or 'log2' not in ngs_df_case.columns or 'absolute_start' not in ngs_df_case.columns:
        return

    log2_values = pd.to_numeric(ngs_df_case['log2'], errors='coerce')
    abs_pos = ngs_df_case['absolute_start']

    # Calculate alpha based on z-score within the case
    alphas = pd.Series(0.5, index=ngs_df_case.index)  # Default alpha
    if not log2_values.isna().all():
        # Use nan_policy='omit' to handle potential NaNs after coercion
        abs_z = np.abs(zscore(log2_values, nan_policy='omit'))
        # Clip z-score effect, apply power for emphasis, ensure min alpha
        calculated_alphas = np.clip(abs_z / 3, 0.1, 1.0) ** 1.75
        # Align calculated alphas back to the original DataFrame index, fill missing with min alpha
        alphas = pd.Series(calculated_alphas, index=log2_values.dropna().index).reindex(ngs_df_case.index).fillna(0.1)

    # Plot points colored by arm classification
    if 'segment_arm' in ngs_df_case.columns:
        unique_arms = ngs_df_case['segment_arm'].dropna().unique()
        if not unique_arms.size > 0:  # No valid arms found
            ax.scatter(abs_pos, log2_values, color=arm_colors_dict.get(None, 'grey'),
                       alpha=alphas, s=2, label=None)
        else:
            for arm in unique_arms:
                mask = ngs_df_case['segment_arm'] == arm
                if mask.any():
                    ax.scatter(abs_pos[mask], log2_values[mask],
                               c=arm_colors_dict.get(arm, 'grey'),
                               alpha=alphas[mask], s=2,
                               label=f'{arm} arm bins' if arm else 'Unclassified bins')  # Label arms once
            # Plot unclassified points if any
            mask_none = ngs_df_case['segment_arm'].isna()
            if mask_none.any():
                ax.scatter(abs_pos[mask_none], log2_values[mask_none],
                           color=arm_colors_dict.get(None, 'grey'),
                           alpha=alphas[mask_none], s=2, label='Unclassified bins')
    else:  # No arm classification available
        ax.scatter(abs_pos, log2_values, color=arm_colors_dict.get(None, 'grey'),
                   alpha=alphas, s=2, label='NGS bins')


def _plot_arm_means(ax, df_aggregated_case, color, linestyle, linewidth, label_text):
    """Plots aggregated arm-level log2 means as horizontal lines."""
    if df_aggregated_case.empty or not all(
            c in df_aggregated_case.columns for c in ['arm_absolute_start', 'arm_absolute_end', 'log2']):
        return

    # Plot each arm segment
    for idx, row in df_aggregated_case.iterrows():
        if pd.notna(row['log2']) and pd.notna(row['arm_absolute_start']) and pd.notna(row['arm_absolute_end']):
            ax.plot(
                [row['arm_absolute_start'], row['arm_absolute_end']],
                [row['log2'], row['log2']],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label_text if idx == 0 else None  # Label only the first segment to avoid duplicates
            )


def _plot_gene_labels(ax, gene_reps_case):
    """Plots gene annotations as points and text labels."""
    if gene_reps_case.empty or not all(c in gene_reps_case.columns for c in ['absolute_position', 'log2', 'gene']):
        return

    valid_gene_reps = gene_reps_case.dropna(subset=['absolute_position', 'log2', 'gene'])
    if valid_gene_reps.empty: return

    # Plot black dots for gene locations
    ax.scatter(
        valid_gene_reps['absolute_position'],
        valid_gene_reps['log2'],
        color='black', s=4, label='genes'  # Updated label
    )

    # Add text labels for each gene
    for _, row in valid_gene_reps.iterrows():
        y, name = row['log2'], row['gene']
        x_pos = row['absolute_position']

        offset = 0.25
        vertical_alignment = 'bottom' if y >= 0 else 'top'  # Place above for pos, below for neg
        y_text = y + offset if y >= 0 else y - offset

        ax.text(
            x_pos,
            y_text,
            name,
            fontsize=9,
            ha='center',
            va=vertical_alignment,
            rotation=90
        )


def draw_cnv_plot(case_identifier, df_ngs_case, df_ngs_agg, df_gene_reps,
                  df_chroms, output_dir, output_filename):
    """Generates and saves the CNV plot for a single case."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting components
    _plot_ngs_scatter(ax, df_ngs_case, ARM_COLORS)
    _plot_arm_means(ax, df_ngs_agg, 'darkviolet', '-', 1.5, 'arm mean copy number')
    _plot_gene_labels(ax, df_gene_reps)

    # Axis lines and ticks
    ax.axhline(0, color='grey', linewidth=0.5)
    if not df_chroms.empty and all(
            c in df_chroms.columns for c in ['chromosome_absolute_start', 'length', 'chromosome']):
        chrom_starts = df_chroms['chromosome_absolute_start']
        ax.vlines(chrom_starts, -PLOT_Y_LIM, PLOT_Y_LIM,
                  color='grey', linestyle='--', linewidth=0.5)

        mids = chrom_starts + df_chroms['length'] / 2
        ax.set_xticks(mids)
        ax.set_xticklabels(df_chroms['chromosome'].astype(str), rotation=90)

    xlim_max = df_chroms['chromosome_absolute_start'].iloc[-1] + df_chroms['length'].iloc[-1]

    ax.set(
        xlim=(0, xlim_max),
        ylim=(-PLOT_Y_LIM, PLOT_Y_LIM),
        xlabel='genomic position by chromosome',
        ylabel='copy number deviation (log2)'
    )

    plt.suptitle(f'CNV profile for {case_identifier}', fontsize=14, y=0.9)
    ax.grid(False)
    ax.legend().set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent title overlap

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / (output_filename if output_filename else f"cnv_plot_{case_identifier.replace('/', '_')}.png")
    try:
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {fname}")
    except Exception as e:
        print(f"Error saving plot {fname}: {e}", file=sys.stderr)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CNV plot from a single NGS .cnr file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("cnr_file", type=Path, help="Path to the input .cnr file.")
    parser.add_argument(
        "-c", "--cytoband", type=Path, default=DEFAULT_CYTOBAND_PATH,
        help="Path to the cytoband definition file (e.g., cytoBand.txt)."
    )
    parser.add_argument(
        "-g", "--genes", type=Path, default=DEFAULT_RELEVANT_GENES_PATH,
        help="Path to the relevant genes definition file (CSV)."
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the output plot."
    )
    parser.add_argument(
        "-f", "--output-filename", type=Path, default=None,
        help="Name of the output plot file (default: cnv_plot_<CASE-ID>.png)."
    )
    parser.add_argument(
        "--case-id", type=str, default=None,
        help="Optional case identifier (default: derived from .cnr filename)."
    )
    parser.add_argument(
        "--gene_annotation_padding", type=int, default=GENE_ANNOTATION_PADDING,
        help="Adjust the interval that is used to annotate genes. Larger intervals may account for measurement inaccuracies. Use 0 for original gene width."
    )

    args = parser.parse_args()

    print("Loading static data...")
    cytoband_df = load_and_preprocess_cytoband(args.cytoband)
    genes_df = load_and_preprocess_relevant_genes(args.genes)

    print("Preprocessing static data...")
    chroms_df, chrom_map = calculate_chromosome_features(cytoband_df)
    arm_map_df = create_arm_mapping_df(cytoband_df, chrom_map)
    arm_lookup_table = create_arm_lookup_table(arm_map_df)
    gene_trees = build_gene_interval_trees(genes_df)

    if chroms_df.empty or arm_lookup_table.empty:
        print("Error: Failed to process essential cytoband features. Exiting.", file=sys.stderr)
        sys.exit(1)

    case_identifier = args.case_id if args.case_id else extract_case_identifier(args.cnr_file)
    print(f"Processing .cnr file for case: {case_identifier}...")
    ngs_raw_df = load_ngs_cnr_file(args.cnr_file, case_identifier)

    if ngs_raw_df is None or ngs_raw_df.empty:
        print(f"Error: Failed to load or process valid data from {args.cnr_file}. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("Annotating and aggregating NGS data...")
    ngs_processed_df = prep_ngs_for_plotting(ngs_raw_df, chrom_map)
    ngs_processed_df = add_segment_arm_classification(ngs_processed_df, arm_lookup_table)
    ngs_processed_df = annotate_segments_with_genes(ngs_processed_df, gene_trees, args.gene_annotation_padding)

    ngs_arm_aggregated = aggregate_ngs_to_arms(ngs_processed_df, chroms_df)
    ngs_gene_reps = aggregate_data_by_gene(ngs_processed_df)

    print(f"NGS Bins: {len(ngs_processed_df)}")
    print(f"Aggregated Arms: {len(ngs_arm_aggregated)}")
    print(f"Gene Representatives: {len(ngs_gene_reps)}")

    print(f"Generating CNV plot for {case_identifier}...")
    draw_cnv_plot(
        case_identifier,
        ngs_processed_df,  # Points for scatter
        ngs_arm_aggregated,  # Lines for arm means
        ngs_gene_reps,  # Points/Labels for genes
        chroms_df,  # Chromosome boundaries/labels
        args.output_dir,
        args.output_filename
    )

    print("Done.")


if __name__ == "__main__":
    main()
