import argparse

import train as T


DEFAULT_INIT = "ckpt.pt"
DEFAULT_DATA = "data_long"

LEARNING_RATE = 5e-5
WARMUP_STEPS = 1000
MAX_STEPS = 50000
BATCH_SIZE = 32
SAVE_EVERY = 5000
EVAL_EVERY = 500
CHECKPOINT_DIR = "checkpoints_finetune"
GRAPH_DIR = "graphs_finetune"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA)
    parser.add_argument("--init_from", type=str, default=DEFAULT_INIT)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--graph_dir", type=str, default=GRAPH_DIR)
    parser.add_argument("--predicted_variance", action="store_true")
    args = parser.parse_args()

    T.LEARNING_RATE = args.lr
    T.WARMUP_STEPS = args.warmup
    T.MAX_STEPS = args.max_steps
    T.BATCH_SIZE = args.batch_size
    T.SAVE_EVERY = args.save_every
    T.EVAL_EVERY = args.eval_every
    T.CHECKPOINT_DIR = args.checkpoint_dir
    T.GRAPH_DIR = args.graph_dir
    T.EMBED_PREDICTED = args.predicted_variance

    print(f"finetune: init_from={args.init_from} data={args.data_dir}")
    print(f"lr={args.lr} warmup={args.warmup} max_steps={args.max_steps} "
          f"batch_size={args.batch_size}")
    print(f"checkpoints -> {args.checkpoint_dir}\n")

    T.train(args.data_dir, init_from=args.init_from)


if __name__ == "__main__":
    main()