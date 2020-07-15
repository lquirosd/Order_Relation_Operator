#!/bin/bash
if [ 3 == 4 ]; then
#-----------------------------------------------------------------------------#
#---- Get OHG dataset
#-----------------------------------------------------------------------------#
cd data/OHG/
#--- size approx 16.3GB
wget https://zenodo.org/record/1322666/files/OHG.tar.gz
tar -xvf OHG.tar.gz 
rm OHG.tar.gz
mkdir data
mv b0* data/.

mkdir train val test
cd train
for f in $(<../train.lst); do
    ln -s ../data/b0*/${f}.tif .
    ln -s ../data/b0*/page/${f}.xml .
done

cd ../val
for f in $(<../val.lst); do
    ln -s ../data/b0*/${f}.tif .
    ln -s ../data/b0*/page/${f}.xml .
done

cd ../test
for f in $(<../test.lst); do
    ln -s ../data/b0*/${f}.tif .
    ln -s ../data/b0*/page/${f}.xml .
done

cd ../../..
fi
#-----------------------------------------------------------------------------#
#---- FCR dataset has to be requested to the owners, hence no automatic 
#     download is available, set-up equires data/FCR/data/ to exists
#-----------------------------------------------------------------------------#
cd data/FCR/
#--- size approx. 1.2GB
wget https://zenodo.org/record/3945088/files/FCR_500.tar.gz
tar -xvf FCR_500.tar.gz
rm FCR_500.tar.gz
mv FCR_500/data .

mkdir train val test
cd train
for f in $(<../train.lst); do
    ln -s ../data/${f}* .
    ln -s ../data/page/${f}* .
done

cd ../val
for f in $(<../val.lst); do
    ln -s ../data/${f}* .
    ln -s ../data/page/${f}* .
done

cd ../test
for f in $(<../test.lst); do
    ln -s ../data/${f}* .
    ln -s ../data/page/${f}* .
done

cd ../../../
exit
#-----------------------------------------------------------------------------#
#---- Get ABP dataset
#-----------------------------------------------------------------------------#
cd data/ABP
#--- size approx 590MB
wget https://zenodo.org/record/1243098/files/READ_ABP_TABLE.zip
unzip READ_ABP_TABLE.zip
rm READ_ABP_TABLE.zip
mv READ_ABP_TABLE/ data

mkdir train val test
cd train
for f in $(<../train.lst); do
    ln -s ../data/dataset111/img/${f}.* .
    ln -s ../data/dataset111/xml/${f}.xml .
done

cd ../val
for f in $(<../val.lst); do
    ln -s ../data/dataset111/img/${f}.* .
    ln -s ../data/dataset111/xml/${f}.xml .
done

cd ../test
for f in $(<../test.lst); do
    ln -s ../data/dataset111/img/${f}.* .
    ln -s ../data/dataset111/xml/${f}.xml .
done

cd ../../../

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

