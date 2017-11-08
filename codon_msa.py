'''Codon multiple sequence alignment.

Takes in a file of nucleotide sequences, translates, aligns, and returns both a
nucleotide and protein alignment. Default aligner is mafft --auto. Output
files will be in FASTA format. If you want something else, you're on your own.

Code stolen and modified from SJ Spielberg,
[thanks](https://github.com/sjspielman/grabbag/blob/master/align_pal2nal.py)

python ~/code/fluprint/scripts/codon_msa.py example.fa --align 'linsi' --align_option=''

Usage:
  codon_msa.py <infile> [--outfile_pr=<outfile_pr>]
    [--outfile_nt=<outfile_nt>] [--align=<aln_exec>] [--align_option=<options>]
  codon_msa.py (-h | --help)
  codon_msa.py --version

Arguments:
    infile                      Sequences in format {fasta, phylip, nexus}.

Options:
  -h --help                     Show this screen.
  --version                     Show version.
  --outfile_pr=<outfile_pr>     Out: protein alignment [default: ./protein.fa].
  --outfile_nt=<outfile_nt>     Out: codon alignment [default: ./codon.fa].
  --align=<aln_exec>            Any mafft algorithm [default: mafft --auto].
  --align_option=<options>      Optional mafft parameters [default: --quiet].
'''

import os
# import sys
# import argparse
import subprocess

from Bio import Seq
from Bio import SeqIO
from Bio.Alphabet import generic_dna
from docopt import docopt


if __name__ == '__main__':
    arguments = docopt(
        __doc__, version='codon msa 0.1dev')
    # print(arguments)


infile = arguments['<infile>']
aln_exec = arguments['--align']
options = arguments['--align_option']
outfile_pr = arguments['--outfile_pr']
outfile_nt = arguments['--outfile_nt']


def parse_translate_infile(infile):
    '''
    Read in input sequence file, which can be in either fasta, phylip, or
    nexus. Translate nucleotide sequences to protein sequences.

    Returns two dictionaries - one with nucleotide seqs, one with protein seqs.
    All unaligned.
    '''
    record_dict = {}
    try:
        records = list(SeqIO.parse(infile, 'fasta'))
    except:
        try:
            records = list(SeqIO.parse(infile, 'phylip-relaxed'))
        except:
            try:
                records = list(SeqIO.parse(infile, 'nexus'))
            except:
                raise AssertionError('''
                    '\n\nCannot read in input sequence file. Verify that it
                    is either in FASTA, phylip, or nexus format.''')
    for rec in records:
        record_dict[str(rec.id)] = str(rec.seq)
    return translate_records(record_dict)


def translate_records(nuc_dict):
    '''If stop codon at last codon, remove it from the nucleotide sequence.'''
    prot_dict = {}
    for entry in nuc_dict:
        nuc_dict[entry] = nuc_dict[entry].replace('-', '')
        # Remove any possible gaps in nucleotide sequence
        nucseq = nuc_dict[entry]
        # print(nucseq)
        assert(len(nucseq) % 3 == 0), 'nt not multiple of three'

        prot_seq = ''
        for i in range(0, len(nucseq), 3):
            codon = nucseq[i:i+3]
            try:
                amino = str(Seq.Seq(codon, generic_dna).translate())
            except:
                raise AssertionError(
                    '\n\nCould not translate input nucleotide codon, quit.')
            if amino == '*':
                # If stop codon is the last codon, just remove it
                if i == len(nucseq)-3:
                    nuc_dict[entry] = nuc_dict[entry][:-3]
                else:
                    # print(nucseq)
                    # print(codon)
                    # continue
                    raise AssertionError(
                        '\n\n You have internal stop codons, quitting.')
            else:
                prot_seq += amino
        prot_dict[entry] = prot_seq
    return nuc_dict, prot_dict


def align_seqs(aln_exec, options, prot_dict, infile, outfile_pr):
    with open(infile, 'w') as inf:
        for rec in prot_dict:
            inf.write('>' + str(rec) + '\n' + str(prot_dict[rec]) + '\n')
    run_align = subprocess.call(
        aln_exec + ' ' + options + ' ' + infile + '>' + outfile_pr, shell=True
        )
    assert(run_align == 0), 'Could not perform mafft alignment.'
    return list(SeqIO.parse(outfile_pr, 'fasta'))


def back_translate(protseq, nucseq):
    '''Back translate an individual sequence'''
    nucaln = ''
    start = 0
    end = 3
    for amino in protseq:
        if amino == '-':
            codon = '---'
        else:
            codon = nucseq[start:end]
            start += 3
            end += 3
        nucaln += codon
    assert(len(protseq)*3 == len(nucaln)), 'Back-translation failed.'
    return nucaln


def pal_to_nal(aln_records, nuc_dict, outfile_pr, outfile_nt):
    protf = open(outfile_pr, 'w')
    nucf = open(outfile_nt,  'w')
    for protrec in aln_records:
        id = str(protrec.id)
        aln_nuc = back_translate(
            str(protrec.seq), nuc_dict[id]
            )
        protf.write('>' + id + '\n' + str(protrec.seq) + '\n')
        nucf.write('>' + id + '\n' + aln_nuc + '\n')
    protf.close()
    nucf.close()


def main():
    # Read in sequences and translate to protein
    print('Reading input sequences')
    nuc_dict, prot_dict = parse_translate_infile(infile)

    # Align protein sequences
    print('Creating protein alignment')
    aln_records = align_seqs(
        aln_exec, options, prot_dict, 'in.fasta', 'out.fasta')

    # Back-translate protein alignment to nucleotide alignment
    print('Back-translating protein alignment to a nucleotide alignment')
    pal_to_nal(aln_records, nuc_dict, outfile_pr, outfile_nt)

    # clean up temp files
    os.remove('out.fasta')
    os.remove('in.fasta')

    outstring = '\nComplete! Your final AA alignment is in ' + \
        str(outfile_pr) + ' and your final nucleotide alignment is in ' + \
        str(outfile_nt) + '\n'
    print(outstring)

main()
