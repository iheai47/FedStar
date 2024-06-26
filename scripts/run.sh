python exps/main_multiDS.py --repeat 1 --data_group 'chem' --seed 1 --alg selftrain
python exps/main_multiDS.py --repeat 2 --data_group 'chem' --seed 2 --alg selftrain
python exps/main_multiDS.py --repeat 3 --data_group 'chem' --seed 3 --alg selftrain
python exps/main_multiDS.py --repeat 4 --data_group 'chem' --seed 4 --alg selftrain
python exps/main_multiDS.py --repeat 5 --data_group 'chem' --seed 5 --alg selftrain

python exps/main_multiDS.py --repeat 1 --data_group 'chem' --seed 1 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 2 --data_group 'chem' --seed 2 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 3 --data_group 'chem' --seed 3 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 4 --data_group 'chem' --seed 4 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 5 --data_group 'chem' --seed 5 --alg fedstar --type_init 'rw_dg'

python exps/main_multiDS.py --repeat 1 --data_group 'biochem' --seed 1 --alg selftrain
python exps/main_multiDS.py --repeat 2 --data_group 'biochem' --seed 2 --alg selftrain
python exps/main_multiDS.py --repeat 3 --data_group 'biochem' --seed 3 --alg selftrain
python exps/main_multiDS.py --repeat 4 --data_group 'biochem' --seed 4 --alg selftrain
python exps/main_multiDS.py --repeat 5 --data_group 'biochem' --seed 5 --alg selftrain

python exps/main_multiDS.py --repeat 1 --data_group 'biochem' --seed 1 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 2 --data_group 'biochem' --seed 2 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 3 --data_group 'biochem' --seed 3 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 4 --data_group 'biochem' --seed 4 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 5 --data_group 'biochem' --seed 5 --alg fedstar --type_init 'rw_dg'

python exps/main_multiDS.py --repeat 1 --data_group 'biochemsn' --seed 1 --alg selftrain
python exps/main_multiDS.py --repeat 2 --data_group 'biochemsn' --seed 2 --alg selftrain
python exps/main_multiDS.py --repeat 3 --data_group 'biochemsn' --seed 3 --alg selftrain
python exps/main_multiDS.py --repeat 4 --data_group 'biochemsn' --seed 4 --alg selftrain
python exps/main_multiDS.py --repeat 5 --data_group 'biochemsn' --seed 5 --alg selftrain

python exps/main_multiDS.py --repeat 1 --data_group 'biochemsn' --seed 1 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 2 --data_group 'biochemsn' --seed 2 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 3 --data_group 'biochemsn' --seed 3 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 4 --data_group 'biochemsn' --seed 4 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 5 --data_group 'biochemsn' --seed 5 --alg fedstar --type_init 'rw_dg'

python exps/main_multiDS.py --repeat 1 --data_group 'biosncv' --seed 1 --alg selftrain
python exps/main_multiDS.py --repeat 2 --data_group 'biosncv' --seed 2 --alg selftrain
python exps/main_multiDS.py --repeat 3 --data_group 'biosncv' --seed 3 --alg selftrain
python exps/main_multiDS.py --repeat 4 --data_group 'biosncv' --seed 4 --alg selftrain
python exps/main_multiDS.py --repeat 5 --data_group 'biosncv' --seed 5 --alg selftrain

python exps/main_multiDS.py --repeat 1 --data_group 'biosncv' --seed 1 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 2 --data_group 'biosncv' --seed 2 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 3 --data_group 'biosncv' --seed 3 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 4 --data_group 'biosncv' --seed 4 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 5 --data_group 'biosncv' --seed 5 --alg fedstar --type_init 'rw_dg'

python exps/main_multiDS.py --repeat 1 --data_group 'chem' --seed 1 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 2 --data_group 'chem' --seed 2 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 3 --data_group 'chem' --seed 3 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 4 --data_group 'chem' --seed 4 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 5 --data_group 'chem' --seed 5 --alg fedstar --type_init 'rw_ds'
python exps/aggregateResults.py --data_group 'chem'

python exps/main_multiDS.py --repeat 1 --data_group 'biochem' --seed 1 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 2 --data_group 'biochem' --seed 2 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 3 --data_group 'biochem' --seed 3 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 4 --data_group 'biochem' --seed 4 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 5 --data_group 'biochem' --seed 5 --alg fedstar --type_init 'rw_ds'
python exps/aggregateResults.py --data_group 'biochem'

python exps/main_multiDS.py --repeat 1 --data_group 'biochemsn' --seed 1 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 2 --data_group 'biochemsn' --seed 2 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 3 --data_group 'biochemsn' --seed 3 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 4 --data_group 'biochemsn' --seed 4 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 5 --data_group 'biochemsn' --seed 5 --alg fedstar --type_init 'rw_ds'
python exps/aggregateResults.py --data_group 'biochemsn'

python exps/main_multiDS.py --repeat 1 --data_group 'biosncv' --seed 1 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 2 --data_group 'biosncv' --seed 2 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 3 --data_group 'biosncv' --seed 3 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 4 --data_group 'biosncv' --seed 4 --alg fedstar --type_init 'rw_ds'
python exps/main_multiDS.py --repeat 5 --data_group 'biosncv' --seed 5 --alg fedstar --type_init 'rw_ds'
python exps/aggregateResults.py --data_group 'biosncv'

python exps/aggregateResults.py --data_group 'chem'
python exps/aggregateResults.py --data_group 'biochem'
python exps/aggregateResults.py --data_group 'biochemsn'
python exps/aggregateResults.py --data_group 'biosncv'
