
import argparse
import os
import subprocess as sp
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check scores')
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--data_path', type=str, default="few-shot-images/AnimalFace-cat/")
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--end_iter', type=int, default=5)
    parser.add_argument('--im_size', type=int, default=1024)
    parser.add_argument('--no_gen', action='store_true')
    parser.set_defaults(gen=False)
    args = parser.parse_args()
    output_file = open(os.path.join(args.root, 'results.txt'),'a')
    output_file.write(f"----------------------------------------------\n")
    if not args.no_gen:
        eval_args = ['python3', 'eval.py', "--batch", "5", "--n_sample", "160", "--im_size", str(args.im_size), "--root", args.root, "--start_iter", str(args.start_iter), "--end_iter", str(args.end_iter)]
        output_file.write(' '.join(eval_args + ["\n"]))
        sp.check_call(eval_args)
    else:
        output_file.write("No gen\n")
    for epoch in [10000*i for i in range(args.start_iter, args.end_iter+1)]:
        out = sp.check_output(['python3', 'benchmark.py', "--data_path", os.path.join("../",args.data_path), "--img_path", os.path.join("../", args.root, "eval_%d"%epoch) ], cwd='benchmarking/')
        print(f"Epoch: {epoch}:")
        print(out.decode('utf-8'))
        output_file.write(f"Epoch: {epoch}:\n")
        output_file.write(out.decode('utf-8'))
        output_file.write("\n")
    output_file.close()
    """
    python3 eval.py --batch 5 --n_sample 160  --root --start_iter --end_iter
    
    """
