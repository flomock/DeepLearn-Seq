import numpy as np
from pyfaidx import Fasta
# from skbio import TabularMSA, DNA
from zoo.align import align_deopt

fp_msa = '/Users/pi/data/influenza/mod/msa/NA/subset_NA_wt.mafft.fa'
fp_deopt = '/Users/pi/data/influenza/external/dusan_deoptimized_influenza/H1N1_NA_deoptimized.fa'


# # msa = TabularMSA.read(fp_msa, format='fasta', constructor=DNA, lowercase=True)
# # for i in msa:
# #     if i.metadata['id'] == 'WT':
# #         a = i


# msa = Fasta(fp_msa)
# wt = str(msa['WT']).upper()

# mod = Fasta(fp_deopt)
# mut = str(mod['DL']).upper()
# mut = align_deopt(wt, mut)


# mismatch = np.array([0 if len(set(i)) == 1 else 1 for i in zip(wt, mut)])
# s = pd.Series(mismatch)
# s.to_csv(
#     '/Users/pi/data/influenza/mod/feature_importance/mismatch_DL_NA.csv',
#     index=True, header=None)


msa = Fasta(fp_msa)
wt = str(msa['WT']).upper()

mod = Fasta(fp_deopt)
path = '/Users/pi/data/influenza/mod/feature_importance/'
for tag in 'DL DM HS OG PD H HD'.split(' '):

    mut = str(mod[tag]).upper()
    mut = align_deopt(wt, mut)
    mismatch = np.array(
        [0 if len(set(i)) == 1 else 1 for i in zip(wt, mut)]
        )
    s = pd.Series(mismatch)
    s.to_csv(
        path + 'mismatch_' + tag + '_NA.csv',
        index=True, header=None)


'''
get cumulative feature importance for modified positions (is the effect
on virus replication additive?)
'''

...


''' R
# Plot entropy and feature importance for NA protein.

library(ggplot2)
library(dplyr)


fp_f <- '/Users/pi/data/influenza/mod/feature_importance/fi_NA.csv'
fp_h <- '/Users/pi/data/influenza/mod/feature_importance/entropy_NA.csv'
fp_m <- '/Users/pi/data/influenza/mod/feature_importance/mismatch_HD_NA.csv'

df_h <- read.table(fp_h, header=F, stringsAsFactors=F, sep=',')
names(df_h) <- c('pos', 'h')
df_f <- read.table(fp_f, header=F, stringsAsFactors=F, sep=',')
names(df_f) <- c('nt', 'pos', 'fi')
df_m <- read.table(fp_m, header=F, stringsAsFactors=F, sep=',')
 names(df_m) <- c('pos', 'mismatch')


df <- inner_join(df_h, df_f, by='pos')
df <- inner_join(df, df_m, by='pos')


# x: position, y: feature importance, facet: nt, color: mismatch


v <-
ggplot(
        df[df$fi > 0.0002,],
        aes(x=pos, y=sqrt(fi), color=as.factor(mismatch))
        ) +
    geom_point(size=0.2) +
    theme_default() +
    facet_wrap(~ nt, scale='fixed', ncol=1) +
    xlab('i') +
    ylab(expression(sqrt(F))) +
    scale_color_manual(
        values=c('#DCBCBC', '#8F2727'),
        guide = guide_legend(title = 'modified')
        )


fp_out = '~/projects/influenza/img/feature_importance/NA_fi_deopt_HD.pdf'
ggsave(fp_out, v, height=16, width=7, unit='cm')


# x: position, y: entropy, color: mismatch


w <-
ggplot(
        df[df$nt == 'A',],
        aes(x=pos, y=h, color=as.factor(mismatch))
        ) +
    geom_point(size=0.2) +
    theme_default() +
    xlab('i') +
    ylab('H') +
    scale_color_manual(
        values=c('#DCBCBC', '#8F2727'),
        guide = guide_legend(title = 'modified')
        ) +
    ggtitle('HD')

fp_out = '~/projects/influenza/img/feature_importance/NA_h_deopt_HD.pdf'
ggsave(fp_out, w, height=5, width=7, unit='cm')
'''


'''
Learn only on the changed positions (the set of all modifications).
'''

np.where(mismatch)












