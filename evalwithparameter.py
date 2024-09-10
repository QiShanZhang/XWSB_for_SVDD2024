# run_with_different_args.py
import subprocess
import sys



# for args in args_list:
#     print(args)
#     subprocess.run(["python", "main_ijcai2024_DF.py"] + args)
#     # print([args[-1][14::], './keys eval'])
#     # subprocess.run(["python", "evaluate_2021_DF.py"] + [args[-1][14::], './keys', 'eval'])
#     break

# --base_dir ./database --model_path logs/rawnet/20240608-162231/checkpoints/best_model_epoch_3.pth --gpu 0 --encoder rawnet --batch_size 40
subprocess.run(["python", "eval.py"] + ['--base_dir','./databaseextention','--model_path',
                                        'logs/rawnet/20240605-000811epoch21-3.26/checkpoints/model_21_EER_0.016918828633939294.pt',
                                        '--gpu', '0' ,'--encoder', 'rawnet', '--batch_size','5', '--output_path', 'logs/rawnet/wavlm/'])
#python DF_eval_with_different_arg.py
