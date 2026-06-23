# Due to limited machine utility, all models are forced on dim=128 and tuned accordingly

GPU=1
SEED=1

bash run_complex.sh train ComplEx FB15k-237 $GPU $SEED 1024 256 128 200.0 0.5 0.001 100000 16 0.000005 -de -dr -seed $SEED
bash run_complex.sh train ComplEx wn18rr $GPU $SEED 512 1024 128 200.0 0.0 0.002 80000 8 0.000001 -de -dr -seed $SEED
bash run_complex.sh train ComplEx YAGO3-10 $GPU $SEED 512 1024 128 200.0 0.25 0.001 150000 8 0.0000005 -de -dr -seed $SEED
bash run_complex.sh train ComplEx family $GPU $SEED 512 128 128 200.0 0.0 0.003 100000 16 0.0000005 -de -dr -seed $SEED
bash run_complex.sh train ComplEx umls $GPU $SEED 128 256 128 200.0 0.0 0.001 100000 16 0.0001 -de -dr -seed $SEED
bash run_complex.sh train ComplEx kinship $GPU $SEED 128 128 128 200.0 0.0 0.0001 100000 16 0.0001 -de -dr -seed $SEED

bash run.sh train RotatE FB15k-237 $GPU $SEED 1024 256 128 6.0 1.0 0.00005 100000 16 -de -seed -seed $SEED
bash run.sh train RotatE wn18rr $GPU $SEED 512 1024 128 3 1.0 0.00005 80000 8 -de -seed $SEED
bash run.sh train RotatE YAGO3-10 $GPU $SEED 1024 400 128 24 0.75 0.0002 150000 4 -de -seed $SEED
bash run.sh train RotatE family $GPU $SEED 512 128 128 24 0.0 0.003 100000 8 -de -seed $SEED
bash run.sh train RotatE umls $GPU $SEED 256 256 128 6 0.25 0.001 100000 8 -de -seed $SEED
bash run.sh train RotatE kinship $GPU $SEED 256 512 128 6.0 0.0 0.003 100000 8 -de -seed $SEED