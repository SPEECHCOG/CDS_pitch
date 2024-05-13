# CDS_pitch

This repository contains the instructions and scripts to replicate experiments reported in:

Daniil Kocharov and Okko Räsänen: Age-dependent intonational changes in child-directed speech. // Proc. Speech Prosody 2023, Leiden, Netherlands. (LINK)

# Experimental Material

The research is based on publicly available Providence Corpus (Demuth, K., Culbertson, J. & Alter, J. 2006. Word-minimality, epenthesis, and coda licensing in the acquisition of English. Language & Speech, 49, 137–174). The data are available in the CHILDES database (https://phon.talkbank.org/access/Eng-NA/Providence.html).

For the current research, a dataset of all the utterances from the Providence corpus, provided by the authors of BabySLM benchmark (https://github.com/MarvinLvn/BabySLM).

Utterances containing phonological fragments (e.g. &, #) or unintelligible speech (e.g., annotated as &, xx, or yy) were excluded from further analysis (see CHAT transcription convention in https://talkbank.org/manuals/CHAT.html).

# Content
- `calculate_opensmile_features.py`: calculation of prosodic features by means of openSMILE.
- `providence_files.csv`: a list of Providence files used in the current research.
- `prosodic_analysis.py`: the main code for analysis of age-dependent speech melody.
- `annotation_utils.py`: classes for processing Praat annotation.

# Main dependencies

- openSMILE (https://github.com/audeering/opensmile)
- matplotlib
- numpy
- pandas
- scipy
- tqdm

# Instructions
The experimental data (speech and its transcripts) should be aligned. It could be done by means of Montreal Forced Aligner (https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) or online WebMAUS tool (https://clarin.phonetik.uni-muenchen.de/BASWebServices/interface/WebMAUSBasic).

The prosodic features should be calculated using `calculate_opensmile_features.py`.

The feature extraction and statistical analysis is produced by `prosodic_analysis.py`. There are a number of parameters that might be specified, see description of command line parameters in the code.
