#------------------------------------------------------------------------------#
#---- OHG
#------------------------------------------------------------------------------#

mkdir -p results/{Classic,MLP,RDF}/OHG/
#--- MLP

# Train and compute F(s,t)
for i in {0..9}; do
    python -u src/MLP.py \
        results/MLP/OHG/ \
        data/OHG/train/ \
        data/OHG/val/ \
        data/OHG/test/ \
        '$pag $tip $par $not $nop $pac' \
        exp_${i} 1 0 | tee results/MLP/OHG/train_exp_${i}.log; 
done
# get averaged F(s,t) across experiments
grep "Test " results/MLP/OHG/train_exp_* | awk '{a+=$5}END{print a/NR}'
# get K(s,t) a-posteriori
for f  in {0..9}; do 
    python src/ktd_post.py \
        results/MLP/OHG/exp_${f}/test/ ;
done | awk '{a+=$3; b+=$5; print $0}END{print a/NR " " b/NR}'

#--- RDF

for f in {0..9}; do 
    python -u src/RDF.py \
        results/RDF/OHG/ \
        data/OHG/train/ \
        data/OHG/val/ \
        data/OHG/test/ \
        '$pag $tip $par $not $nop $pac' \
        10 10 exp_${f} | tee results/RDF/OHG/train_exp_${f}.log ; 
done 

#--- Classic top-down/left-right

python src/basic_ro.py results/Classic/OHG/ data/OHG/test/processed/test.pickle 6

#------------------------------------------------------------------------------#
#---- FCR
#------------------------------------------------------------------------------#
mkdir -p results/{Classic,MLP,RDF}/FCR/

#--- MLP

# Train and compute F(s,t)
for i in {0..9}; do
    python -u src/MLP.py \
        results/MLP/FCR/ \
        data/FCR/train/ \
        data/FCR/val/ \
        data/FCR/test/ \
        'paragraph paragraph_2 marginalia page-number table table_2' \
        exp_${i} 1 0 | tee results/MLP/FCR/train_exp_${i}.log; 
done
# get averaged F(s,t) across experiments
grep "Test " results/MLP/FCR/train_exp_* | awk '{a+=$5}END{print a/NR}'
# get K(s,t) a-posteriori
for f  in {0..9}; do 
    python src/ktd_post.py \
        results/MLP/FCR/exp_${f}/test/ ;
done | awk '{a+=$3; b+=$5; print $0}END{print a/NR " " b/NR}'

#--- RDF no Data Augmentation

for f in {0..9}; do 
    python -u src/RDF.py \
        results/RDF/FCR/ \
        data/FCR/train/ \
        data/FCR/val/ \
        data/FCR/test/ \
        'paragraph paragraph_2 marginalia page-number table table_2' \
        10 10 exp_${f} | tee results/RDF/FCR/train_exp_${f}.log ; done 

#--- Classic top-down/left-right

python src/basic_ro.py results/RDF/FCR/ data/FCR/test/processed/test.pickle 6

#------------------------------------------------------------------------------#
#---- ABP
#------------------------------------------------------------------------------#

mkdir -p results/{Classic,MLP,RDF}/FCR/

#--- MLP

# Train and compute F(s,t)
for i in {0..9}; do
    python -u src/MLP.py \
        results/MLP/ABP/ \
        data/ABP/train/ \
        data/ABP/val/ \
        data/ABP/test/ \
        'TextRegion TableRegion' \
        exp_${i} 1 1 | tee results/MLP/ABP/train_exp_${i}.log; 
done
# get averaged F(s,t) across experiments
grep "Test " results/MLP/ABP/train_exp_* | awk '{a+=$5}END{print a/NR}'
# get K(s,t) a-posteriori
for f  in {0..9}; do 
    python src/ktd_post.py \
        results/MLP/ABP/exp_${f}/test/ ;
done | awk '{a+=$3; b+=$5; print $0}END{print a/NR " " b/NR}'

#--- RDF

for f in {0..9}; do 
    python -u src/RDF.py \
        results/RDF/ABP/ \
        data/ABP/train/ \
        data/ABP/val/ \
        data/ABP/test/ \
        'TextRegion TableRegion' \
        10 10 exp_${f} | tee results/RDF/ABP/train_exp_${f}.log ; done 

#--- Classic top-down/left-right

python src/basic_ro.py results/Classic/ABP/ data/ABP/test/processed/test.pickle 2

#------------------------------------------------------------------------------#
#---- PLOT some results ...
#------------------------------------------------------------------------------#
#--- MLP-OHG
python src/plot_reading_order.py \
    paper/MLP/OHG/exp_0/test/ \
    data/OHG/test/ \
    data/OHG/test/processed/test.pickle \
    '$pag $tip $par $not $nop $pac' \
    paper/MLP/OHG/exp_0/test/

#--- MLP-FCR
python src/plot_reading_order.py \
    paper/MLP/FCR/exp_0/test/ \
    data/FCR/test/ \
    data/FCR/test/processed/test.pickle \
    'paragraph paragraph_2 marginalia page-number table table_2' \
    paper/MLP/NAF/exp_0/test/

#--- MLP-ABP
python src/plot_reading_order.py \
    paper/MLP/ABP/exp_0/test/ \
    data/ABP/test/ \
    data/ABP/test/processed/test.pickle \
    'TextRegion TableRegion' \
    paper/MLP/ABP/exp_0/test/
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
