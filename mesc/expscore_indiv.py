'''
Compute expression scores and estimate expression cis-heritability from individual-level expression and genotype data
'''

from __future__ import division
import numpy as np
import pandas as pd
import os
import argparse
import collections
import subprocess
import ldscore as ld
import parse as ps
import sys
import gzip
import bz2

class Suppressor(object):
    '''
    Suppresses output from subprocess
    '''
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            pass

    def write(self, x): pass

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def flatten_list(x):
    if isinstance(x, collections.Iterable) and not isinstance(x, basestring):
        return [a for i in x for a in flatten_list(i)]
    else:
        return [x]

def file_len(fname, input_chr, chr_idx):
    '''
    Get number of genes in gene expression file on input chromosome
    '''
    count = 0

    # check compression
    if fname.endswith("gz"):
        anyopen = gzip.open
    elif fname.endswith("bz2"):
        anyopen = bz2.open
    else:
        anyopen = open

    with anyopen(fname) as f:
        for i, l in enumerate(f):
            if i == 0:
                continue
            l = l.split()
            try:
                chr = int(l[chr_idx])
            except:
                continue
            if chr == input_chr:
                count += 1
    return count

def get_eqtl_annot(args, gene_name, phenos, start_bp, end_bp, geno_fname, sample_names, chr, covar):
    '''
    Create cis-region genotype file, estimate eQTL effects sizes using LASSO, and estimate expression cis-heritability using REML
    :return herit: REML-estimated h2cis
    :return herit_se: h2cis SE
    :return herit_p: h2cis p-value
    :return lasso: eQTL effect size estimates from LASSO
    '''
    keep_snp_name = '{}/keep_snps_chr_{}.txt'.format(args.tmp, args.chr)

    FNULL = open(os.devnull, 'w')
    pheno_fname = '{}/{}.pheno'.format(args.tmp, gene_name)
    temp_geno_fname = '{}/{}'.format(args.tmp, gene_name)

    # making temporary phenotype file
    temp_pheno = pd.concat([sample_names, pd.Series(phenos)], axis=1, ignore_index=True)
    temp_pheno.to_csv(pheno_fname, sep='\t', index=False, header=False)

    # for some reason LASSO performs better when covariates are regressed out of phenotype instead of being included in regression
    # REML performs better when covariates are included rather than regressed out
    if covar is not None:
        pheno_covar_fname = '{}/{}_covar.pheno'.format(args.tmp, gene_name)
        phenos_reg = np.array([float(x) for x in phenos])
        res = np.linalg.lstsq(covar.iloc[:, 2:].values, phenos_reg, rcond=None)
        phenos_reg -= np.dot(covar.iloc[:, 2:].values, res[0])
        phenos_reg = phenos_reg.tolist()
        temp_pheno_reg = pd.concat([sample_names, pd.Series(phenos_reg)], axis=1, ignore_index=True)
        temp_pheno_reg.to_csv(pheno_covar_fname, sep='\t', index=False, header=False)

    # create cis-region genotype file
    try:
        subprocess.check_output(
            [args.plink_path, '--bfile', geno_fname, '--pheno', pheno_fname, '--make-bed', '--out', temp_geno_fname,
             '--chr', str(chr), '--from-bp', str(start_bp), '--to-bp', str(end_bp), '--extract', keep_snp_name, '--silent',
             '--allow-no-sex'], stderr=FNULL)

    except subprocess.CalledProcessError:
        subprocess.call('rm {}*'.format(temp_geno_fname), shell=True)
        return 'NO_PLINK'

    # make grm (for REML)
    subprocess.call(
        [args.plink_path, '--bfile', temp_geno_fname, '--allow-no-sex', '--make-grm-bin', '--out', temp_geno_fname, '--silent'])

    # estimate h2cis using REML
    if covar is not None:
        subprocess.call(
            [args.gcta_path, '--grm', temp_geno_fname, '--pheno', pheno_fname, '--qcovar', args.covariates, '--out', temp_geno_fname,
             '--reml', '--reml-no-constrain', '--reml-lrt', '1'], stdout=FNULL, stderr=FNULL)
    else:
        subprocess.call(
            [args.gcta_path, '--grm', temp_geno_fname, '--pheno', pheno_fname, '--out', temp_geno_fname,
             '--reml', '--reml-no-constrain', '--reml-lrt', '1'], stdout=FNULL, stderr=FNULL)

    hsq_fname = '{}.hsq'.format(temp_geno_fname)

    if os.path.exists(hsq_fname):
        hsq = pd.read_csv(hsq_fname, sep='\t')
        herit = hsq.iloc[3, 1]
        herit_se = hsq.iloc[3, 2]
        herit_p = hsq.iloc[8, 1]
    else:
        herit = np.nan
        herit_se = np.nan
        herit_p = np.nan

    if np.isnan(herit) or herit < 0:
        if np.isnan(herit):
            print('Skipping; REML did not converge')
        elif herit < 0:
            print('Skipping; h2cis < 0')
        subprocess.call('rm {}*'.format(temp_geno_fname), shell=True)
        return (herit, herit_se, herit_p, np.nan)

    # estimate causal eQTL effect sizes using LASSO
    if covar is not None:
        subprocess.call(
            [args.plink_path, '--allow-no-sex', '--bfile', temp_geno_fname, '--lasso', str(herit), '--pheno', pheno_covar_fname,
             '--out', temp_geno_fname, '--silent'], stdout=FNULL, stderr=FNULL)
    else:
        subprocess.call(
            [args.plink_path, '--allow-no-sex', '--bfile', temp_geno_fname, '--lasso', str(herit),
             '--out', temp_geno_fname, '--silent'], stdout=FNULL, stderr=FNULL)

    if os.path.exists('{}.lasso'.format(temp_geno_fname)):
        lasso = pd.read_csv('{}.lasso'.format(temp_geno_fname), sep='\t')
    else:
        print('Skipping; LASSO did not converge')
        subprocess.call('rm {}*'.format(temp_geno_fname), shell=True)
        return (herit, herit_se, herit_p, np.nan)

    lasso_weights = lasso['EFFECT'].values
    emp_herit = np.sum(np.square(lasso_weights))
    if not np.isnan(herit):
        if herit <= 0 or emp_herit <= 0:
            bias = 0
        else:
            bias = np.sqrt(herit / emp_herit)
        lasso['CORR_EFFECT'] = lasso_weights * bias
    else:
        lasso['CORR_EFFECT'] = np.nan

    lasso['GENE'] = gene_name
    if lasso.shape[0] == 0:
        lasso = np.nan
    out = (herit, herit_se, herit_p, lasso)
    subprocess.call('rm {}*'.format(temp_geno_fname), shell=True)
    return out

def get_expression_scores(args):
    '''
    Estimate expression scores and expression cis-heritability
    '''
    expmat = args.expression_matrix
    if args.columns:
        columns = args.columns.split(',')
        columns = [int(x)-1 for x in columns]
        if len(columns) != 4:
            raise ValueError('Must specify 4 column indices with --columns')
    else:
        columns = range(4)
    n_genes = file_len(expmat, args.chr, columns[1])
    gene_num = 0

    # making temporary keep snps file (merging args.keep w/ geno_bfile .bim)
    keep_snps = pd.read_csv(args.keep, header=None)
    keep_snps_geno = pd.read_csv(args.geno_bfile + '.bim', header=None, delim_whitespace=True)
    
    # Use pandas merge for better performance with large datasets
    keep_df = pd.DataFrame({'SNP': keep_snps[0]})
    geno_df = pd.DataFrame({'SNP': keep_snps_geno[1]})
    merged = keep_df.merge(geno_df, on='SNP', how='inner')
    keep_snps = pd.DataFrame({0: merged['SNP']})
    
    keep_snps.to_csv('{}/keep_snps_chr_{}.txt'.format(args.tmp, args.chr), header=False, index=False)

    print('Analyzing chromosome {}'.format(args.chr))
    
    # Load gene list filter if provided
    gene_filter = None
    filtered_gene_count = n_genes  # Default to all genes
    if hasattr(args, 'gene_list') and args.gene_list:
        print('Loading gene list from {}'.format(args.gene_list))
        gene_filter = set()
        with open(args.gene_list, 'r') as f:
            for line in f:
                gene_filter.add(line.strip())
        print('Filtering to {} genes from gene list'.format(len(gene_filter)))
        filtered_gene_count = len(gene_filter)  # Use filtered count for display
    
    all_lasso = []
    all_herit = []
    glist = []

    geno_fname = args.exp_bfile
    exp_indivs = pd.read_csv(geno_fname + '.fam', header=None, delim_whitespace=True)

    if args.covariates:
        covar = pd.read_csv(args.covariates, delim_whitespace=True)
    else:
        covar = None

    # check compression
    if expmat.endswith("gz"):
        anyopen = gzip.open
    elif expmat.endswith("bz2"):
        anyopen = bz2.open
    else:
        anyopen = open

    with anyopen(expmat) as f:

        # compute h2cis and estimate LASSO effect sizes for all genes
        for j, line in enumerate(f):
            line = line.split()
            if j == 0:
                sample_names = pd.DataFrame({1: line[columns[3]:]})
                sample_names = sample_names.merge(exp_indivs.iloc[:, [0, 1]], on=1, how='left')
                sample_names = sample_names.iloc[:,[1,0]]
                continue
            gene = line[columns[0]]

            try:
                chr = int(line[columns[1]])
            except:
                continue
            if chr != args.chr:
                continue
            start_bp = max(1, int(line[columns[2]]) - 5e5)
            end_bp = int(line[columns[2]]) + 5e5
            phenos = line[columns[3]:]

            # Skip if gene is not in filter list
            if gene_filter is not None and gene not in gene_filter:
                continue
                
            gene_num += 1
            print('Estimating eQTL effect sizes for gene {} of {}: {}'.format(gene_num, filtered_gene_count, gene))
            if gene in glist:
                print('Skipping; duplicate gene')
                continue

            # some genes have a slash in the name??
            if '/' in gene:
                print('Skipping; "/" in gene name')
                continue

            herit = get_eqtl_annot(args, gene, phenos, start_bp, end_bp, geno_fname, sample_names, chr, covar)

            if herit == 'NO_PLINK':
                print('Skipping; no SNPS around gene')
            else:
                all_herit.append([gene, args.chr, herit[0], herit[1], herit[2]])
                if isinstance(herit[3], pd.DataFrame):
                    all_lasso.append((gene, herit[0], herit[3]))
            glist.append(gene)

    if len(all_lasso) == 0:
        raise ValueError('No weights estimated; something is wrong with input data.')

    # remove keep snps file
    subprocess.call('rm {}/keep_snps_chr_{}.txt'.format(args.tmp, args.chr), shell=True)

    # output h2cis estimates
    all_herit = pd.DataFrame.from_records(all_herit, columns=['Gene', 'Chrom', 'h2cis', 'h2cis_se', 'h2cis_p'])
    all_herit.to_csv('{}.{}.hsq'.format(args.out, args.chr), sep='\t', index=False, float_format='%.5f', na_rep='NA')

    # output LASSO estimates
    lasso_out = pd.concat([x[2] for x in all_lasso])
    lasso_out = lasso_out[['GENE', 'CHR', 'SNP', 'EFFECT']]
    lasso_out.to_csv('{}.{}.lasso'.format(args.out, args.chr), sep='\t', index=False, float_format='%.8f', na_rep='NA')

    # estimate expression scores
    if not args.est_lasso_only:

        # create gene annotation files
        # load genotypes
        sc_geno_fname = args.geno_bfile
        array_indivs = ps.PlinkFAMFile(sc_geno_fname + '.fam')
        array_snps = ps.PlinkBIMFile(sc_geno_fname + '.bim')
        keep_snps_indices = np.where((array_snps.df['CHR'] == args.chr).values & array_snps.df['SNP'].isin(keep_snps[0]).values)[0]
        with Suppressor():
            geno_array = ld.PlinkBEDFile(sc_geno_fname + '.bed', array_indivs.n, array_snps,
                                         keep_snps=keep_snps_indices)
        # SNP indices as dict for fast merging
        snp_indices = dict(zip(geno_array.df[:, 1].tolist(), range(len(geno_array.df))))
        # exclude genes with h2cis < 0 or not converged
        all_lasso_temp = [x for x in all_lasso if not np.isnan(x[1])]
        all_lasso_temp = [x for x in all_lasso_temp if x[1] > 0]

        lasso_herits = [x[1] for x in all_lasso_temp]
        g_annot = np.zeros((len(all_lasso_temp), args.num_bins), dtype=int)
        eqtl_annot = np.zeros((len(geno_array.df), args.num_bins))
        gene_bins = pd.qcut(np.array(lasso_herits), args.num_bins, labels=range(args.num_bins)).astype(int)
        g_bin_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, args.num_bins+1)]
        for j in range(0, len(all_lasso_temp)):
            g_annot[j, gene_bins[j]] = 1
            snp_idx = [snp_indices[x] for x in all_lasso_temp[j][2]['SNP'].tolist()]
            eqtl_annot[snp_idx, gene_bins[j]] += np.square(all_lasso_temp[j][2]['CORR_EFFECT'].values)

        g_annot_final = pd.DataFrame(np.c_[[x[0] for x in all_lasso_temp], g_annot])
        g_annot_final.columns = ['Gene'] + g_bin_names
        g_annot_final.to_csv('{}.{}.gannot.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip')

        matched_herit = all_herit.loc[all_herit['Gene'].isin(g_annot_final['Gene']), 'h2cis'].values
        G = np.sum(g_annot, axis=0)
        ave_cis_herit = np.dot(matched_herit, g_annot) / G

        np.savetxt('{}.{}.G'.format(args.out, args.chr), G.reshape((1, len(G))), fmt='%d')
        np.savetxt('{}.{}.ave_h2cis'.format(args.out, args.chr), ave_cis_herit.reshape((1, len(ave_cis_herit))),
                   fmt="%.5f")

        print('Computing expression scores')

        block_left = ld.getBlockLefts(geno_array.df[:,2], 1e6)

        # estimate expression scores
        res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=eqtl_annot)
        expscore = pd.concat([
            pd.DataFrame(geno_array.df[:, :3]),
            pd.DataFrame(res)], axis=1)
        expscore.columns = geno_array.colnames[:3] + g_bin_names

        # output files
        expscore.to_csv('{}.{}.expscore.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip',
                        float_format='%.5f')

    print('Done chromosome {}'.format(args.chr))
    print('All done!')


def compute_expression_scores_from_lasso(args):
    '''
    Compute expression scores from pre-computed LASSO results
    This allows distributed computation where LASSO estimation is done in chunks
    and then expression scores are computed from the merged results
    
    Parameters:
    args: argparse object with:
        - lasso_files: List of LASSO result files from chunks
        - hsq_files: List of heritability result files from chunks
        - geno_bfile: Genotype file for computing LD scores
        - chr: Chromosome number
        - out: Output prefix
        - num_bins: Number of heritability bins (default 5)
        - keep: SNP list file
    
    Outputs:
        - {out}.{chr}.hsq: Merged heritability estimates
        - {out}.{chr}.lasso: Merged LASSO effects
        - {out}.{chr}.gannot.gz: Gene annotations
        - {out}.{chr}.G: Gene counts per bin
        - {out}.{chr}.ave_h2cis: Average heritability per bin
        - {out}.{chr}.expscore.gz: Expression scores
    '''
    print('Computing expression scores from pre-computed LASSO results')
    print('Reading {} LASSO files and {} heritability files'.format(
        len(args.lasso_files), len(args.hsq_files)))
    
    # Read and merge LASSO results
    lasso_dfs = []
    for lasso_file in args.lasso_files:
        print('Reading LASSO file: {}'.format(lasso_file))
        df = pd.read_csv(lasso_file, sep='\t')
        lasso_dfs.append(df)
    
    lasso_df = pd.concat(lasso_dfs, ignore_index=True)
    print('Merged {} LASSO effects from {} genes'.format(len(lasso_df), lasso_df['GENE'].nunique()))
    
    # Read and merge heritability results
    hsq_dfs = []
    for hsq_file in args.hsq_files:
        print('Reading heritability file: {}'.format(hsq_file))
        df = pd.read_csv(hsq_file, sep='\t')
        hsq_dfs.append(df)
    
    all_herit = pd.concat(hsq_dfs, ignore_index=True)
    print('Merged heritability estimates for {} genes'.format(len(all_herit)))
    
    # Output merged heritability estimates
    all_herit.to_csv('{}.{}.hsq'.format(args.out, args.chr), sep='\t', index=False, float_format='%.5f', na_rep='NA')
    print('Saved merged heritability estimates to {}.{}.hsq'.format(args.out, args.chr))
    
    # Load genotype data for LD score computation
    print('Loading genotype data from {}'.format(args.geno_bfile))
    array_indivs = ps.PlinkFAMFile(args.geno_bfile + '.fam')
    array_snps = ps.PlinkBIMFile(args.geno_bfile + '.bim')
    
    # Filter SNPs using the standard keep file (same as original implementation)
    keep_file = args.keep if hasattr(args, 'keep') else os.path.join(os.path.dirname(__file__), '../data/hm3_snps.txt')
    print('Using SNP list from: {}'.format(keep_file))
    keep_snps = pd.read_csv(keep_file, header=None, delim_whitespace=True)
    keep_snps.columns = ['SNP']
    
    # Get SNP indices
    keep_snps_indices = np.where(
        (array_snps.df['CHR'] == args.chr).values & 
        array_snps.df['SNP'].isin(keep_snps['SNP']).values
    )[0]
    
    print('Using {} SNPs on chromosome {}'.format(len(keep_snps_indices), args.chr))
    
    # Load genotype array with suppressed output
    with Suppressor():
        geno_array = ld.PlinkBEDFile(args.geno_bfile + '.bed', array_indivs.n, array_snps,
                                     keep_snps=keep_snps_indices)
    
    # SNP indices as dict for fast merging
    snp_indices = dict(zip(geno_array.df[:, 1].tolist(), range(len(geno_array.df))))
    
    # Prepare data structure similar to original all_lasso
    # Structure: [(gene, h2cis, lasso_df_with_CORR_EFFECT), ...]
    all_lasso_temp = []
    
    for gene in lasso_df['GENE'].unique():
        # Get h2cis from heritability file
        gene_herit = all_herit[all_herit['Gene'] == gene]
        if len(gene_herit) == 0:
            continue
        
        h2cis = gene_herit['h2cis'].values[0]
        
        # Skip genes with NaN or negative h2cis (matching original)
        if np.isnan(h2cis) or h2cis <= 0:
            continue
        
        # Get LASSO effects for this gene
        gene_lasso = lasso_df[lasso_df['GENE'] == gene].copy()
        if len(gene_lasso) == 0:
            continue
        
        # Calculate empirical heritability and bias correction
        lasso_weights = gene_lasso['EFFECT'].values
        emp_herit = np.sum(np.square(lasso_weights))
        
        if emp_herit <= 0:
            bias = 0
        else:
            bias = np.sqrt(h2cis / emp_herit)
        
        # Add CORR_EFFECT column (bias-corrected effects)
        gene_lasso['CORR_EFFECT'] = lasso_weights * bias
        
        # Add to list in same format as original
        all_lasso_temp.append((gene, h2cis, gene_lasso))
    
    print('{} genes passed filters (positive h2cis with LASSO effects)'.format(len(all_lasso_temp)))
    
    if len(all_lasso_temp) == 0:
        raise ValueError('No genes with positive heritability found')
    
    # Output merged LASSO estimates (format matching original output)
    # Extract just the GENE, CHR, SNP, EFFECT columns
    lasso_output = lasso_df[['GENE', 'CHR', 'SNP', 'EFFECT']]
    lasso_output.to_csv('{}.{}.lasso'.format(args.out, args.chr), sep='\t', index=False, float_format='%.8f', na_rep='NA')
    print('Saved merged LASSO estimates to {}.{}.lasso'.format(args.out, args.chr))
    
    # Extract LASSO heritabilities for binning (use h2cis from REML, not empirical)
    lasso_herits = [x[1] for x in all_lasso_temp]
    
    # Create gene annotation and eQTL annotation matrices
    g_annot = np.zeros((len(all_lasso_temp), args.num_bins), dtype=int)
    eqtl_annot = np.zeros((len(geno_array.df), args.num_bins))
    
    # Bin genes by REML h2cis
    gene_bins = pd.qcut(np.array(lasso_herits), args.num_bins, labels=range(args.num_bins)).astype(int)
    g_bin_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, args.num_bins+1)]
    
    # Fill annotation matrices
    for j in range(len(all_lasso_temp)):
        gene, h2cis, gene_lasso = all_lasso_temp[j]
        
        # Gene annotation
        g_annot[j, gene_bins[j]] = 1
        
        # eQTL annotation - use CORR_EFFECT squared
        snp_idx = [snp_indices[x] for x in gene_lasso['SNP'].tolist()]
        eqtl_annot[snp_idx, gene_bins[j]] += np.square(gene_lasso['CORR_EFFECT'].values)
    
    # Create gene annotation output
    g_annot_final = pd.DataFrame(np.c_[[x[0] for x in all_lasso_temp], g_annot])
    g_annot_final.columns = ['Gene'] + g_bin_names
    g_annot_final.to_csv('{}.{}.gannot.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip')
    print('Saved gene annotations to {}.{}.gannot.gz'.format(args.out, args.chr))
    
    # Calculate average heritability per bin (using REML h2cis)
    matched_herit = all_herit.loc[all_herit['Gene'].isin(g_annot_final['Gene']), 'h2cis'].values
    G = np.sum(g_annot, axis=0)
    ave_cis_herit = np.dot(matched_herit, g_annot) / G
    
    # Save gene counts and average heritabilities
    np.savetxt('{}.{}.G'.format(args.out, args.chr), G.reshape((1, len(G))), fmt='%d')
    np.savetxt('{}.{}.ave_h2cis'.format(args.out, args.chr), ave_cis_herit.reshape((1, len(ave_cis_herit))),
               fmt="%.5f")
    print('Saved gene counts to {}.{}.G'.format(args.out, args.chr))
    print('Saved average heritabilities to {}.{}.ave_h2cis'.format(args.out, args.chr))
    
    # Compute expression scores using LD score regression
    print('Computing expression scores...')
    
    # Get coordinates for LD window
    block_left = ld.getBlockLefts(geno_array.df[:, 2], 1e6)
    
    # Estimate expression scores using built-in method
    res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=eqtl_annot)
    
    # Create output dataframe
    expscore = pd.concat([
        pd.DataFrame(geno_array.df[:, :3]),
        pd.DataFrame(res)], axis=1)
    expscore.columns = geno_array.colnames[:3] + g_bin_names
    
    # Save expression scores
    expscore.to_csv('{}.{}.expscore.gz'.format(args.out, args.chr), sep='\t', index=False, 
                    compression='gzip', float_format='%.5f')
    print('Saved expression scores to {}.{}.expscore.gz'.format(args.out, args.chr))
    
    print('Expression score computation from LASSO files completed!')
    print('Output files:')
    print('  - {}.{}.hsq (merged heritability estimates)'.format(args.out, args.chr))
    print('  - {}.{}.lasso (merged LASSO effects)'.format(args.out, args.chr))
    print('  - {}.{}.gannot.gz (gene annotations)'.format(args.out, args.chr))
    print('  - {}.{}.G (gene counts per bin)'.format(args.out, args.chr))
    print('  - {}.{}.ave_h2cis (average heritability per bin)'.format(args.out, args.chr))
    print('  - {}.{}.expscore.gz (expression scores)'.format(args.out, args.chr))

