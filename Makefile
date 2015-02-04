all: dep env download build_db

dep:
	# https://iqbalnaved.wordpress.com/2013/11/12/installing-scikit-learn-for-virtualenv-python-2-7-5-on-64-bit-ubuntu-13-10/
	# http://docs.scipy.org/doc/numpy/user/install.html#building-with-atlas-support
	sudo apt-get install build-essential python-dev libatlas-dev libatlas3-base libatlas-base-dev liblapack-dev gfortran libpng12-dev libfreetype6-dev
	# python3 virtualenv
	sudo pip3 install -U virtualenv

env:
	virtualenv env
	env/bin/pip install numpy
	env/bin/pip install -r requirements

download:
	env/bin/python -m nltk.downloader -d data punkt
	env/bin/python -m nltk.downloader -d data maxent_treebank_pos_tagger
	wget https://github.com/leebird/legonlp/archive/master.zip -O data/legonlp.zip
	cd data; unzip legonlp.zip
	
build_db:
	-rm -rf data/atom_db/
	-rm -rf data/bigram_db/
	env/bin/python -m src.build_dict data/dict/UMLS/disease_names.txt.gz data/atom_db data/bigram_db
	env/bin/python -m src.build_dict data/dict/UMLS/disease_names_atom.txt.gz data/atom_db data/bigram_db
	env/bin/python -m src.build_dict data/dict/MEDIC/disease_names.txt.gz data/atom_db data/bigram_db

test:
	source env/bin/activate

#python -m crf.build_dict old/UMLS/api/disease_names.txt data/atom_db data/bigram_db
#python -m crf.build_dict old/UMLS/api/disease_names_atom.txt data/atom_db data/bigram_db
#python -m crf.build_dict old/MEDIC/disease_names.txt data/atom_db data/bigram_db

#python -m crf.train corpus/BIO/train.bio model/model
#python -m crf.test corpus/BIO/development.bio model/model

#PYTHONPATH=../legonlp python -m crf.bio_to_ann
#PYTHONPATH=../legonlp python -m crf.evaluate data/result/user_test/ data/result/gold_test/