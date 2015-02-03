# https://iqbalnaved.wordpress.com/2013/11/12/installing-scikit-learn-for-virtualenv-python-2-7-5-on-64-bit-ubuntu-13-10/
sudo apt-get install build-essential python-dev libatlas-dev libatlas3-base liblapack-dev gfortran libpng12-dev libfreetype6-dev

python -m crf.build_dict old/UMLS/api/disease_names.txt data/atom_db data/bigram_db
python -m crf.build_dict old/UMLS/api/disease_names_atom.txt data/atom_db data/bigram_db
python -m crf.build_dict old/MEDIC/disease_names.txt data/atom_db data/bigram_db

python -m crf.train corpus/BIO/train.bio model/model
python -m crf.test corpus/BIO/development.bio model/model