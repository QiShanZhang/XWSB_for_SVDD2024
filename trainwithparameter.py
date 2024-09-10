# run_with_different_args.py
import subprocess
import sys



# for args in args_list:
#     print(args)
#     subprocess.run(["python", "main_ijcai2024_DF.py"] + args)
#     # print([args[-1][14::], './keys eval'])
#     # subprocess.run(["python", "evaluate_2021_DF.py"] + [args[-1][14::], './keys', 'eval'])
#     break

# --base_dir ./database --gpu 0 --encoder rawnet --batch_size 20
subprocess.run(["python", "train.py"] + ['--base_dir','./database', '--gpu', '0' ,'--encoder', 'rawnet', '--batch_size','12'])
#python DF_eval_with_different_arg.py
