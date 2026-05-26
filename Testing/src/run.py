import os

'''
Commands that can be done either independent to model & dataset or case studies.
'''

if False:
    cmd_ls = [
        # f"python .\main.py --data FB15k237 --model RotatE               --fig4",
        # f"python .\main.py --data FB15k237 --model RotatE               --fig5",
        # f"python .\main.py --data FB15k237 --model RotatE               --fig6_a",
        # f"python .\main.py --data FB15k237 --model RotatE               --fig6_bcd",
        # f"python .\main.py --data FB15k237 --model RotatE               --fig7",
        # f"python .\main.py --data FB15k237 --model RotatE               --fig12",
        
        # f"python .\main.py --data FB15k237 --model RotatE RNNLogic      --fig2_obs1",
        # f"python .\main.py --data FB15k237 --model RotatE RNNLogic      --fig2_obs2",
        
        # f"python .\main.py --data wn18rr --model HousE RNNLogic         --fig9",
        # f"python .\main.py --data YAGO3-10 --model pLogicNet HousE      --fig9",
        
        # f"python .\main.py --data FB15k237 --model pLogicNet TuckER     --fig10_top",
        # f"python .\main.py --data umls --model ComplEx RNNLogic         --fig10_top",
        
        # f"python .\main.py --data FB15k237 --model pLogicNet TuckER     --fig10_bottom",
        # f"python .\main.py --data umls --model ComplEx RNNLogic         --fig10_bottom",
    ]
    
    for cmd in cmd_ls:
        print(f"Running: {cmd}")
        os.system(cmd)

##########################################################################################

'''
Commands that require all models. It also iterates over the dataset.
'''

if True:
    datasets = ['FB15k237', 'wn18rr', 'YAGO3-10', 'family', 'umls', 'kinship']
    datasets = ['YAGO3-10', ]
    
    arg_ls = [
        # f"--table4",
        # f"--table5",
        f"--fig11",
    ]
    
    for data in datasets:
        if data in {'YAGO3-10'}: # No results for RNNLogic on YAGO3-10 due to OOM.
            models = ['RotatE', 'ComplEx', 'HousE', 'TuckER', 'pLogicNet']
        else: 
            models = ['RotatE', 'ComplEx', 'HousE', 'TuckER', 'pLogicNet','RNNLogic']
        cmd = f"python main.py --data {data} --model {' '.join(models)} {' '.join(arg_ls)}"

        print(f"Running: {cmd}")
        os.system(cmd)