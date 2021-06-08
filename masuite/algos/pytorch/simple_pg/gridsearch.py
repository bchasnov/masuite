from masuite.algos.pytorch.simple_pg import run

lrs = [float(f'1e-{i}') for i in range(1, 10)]
batch_sizes = [1000*i for i in range(1, 10)]
# epochs = [10*i for i in range(1, 10)]
epochs = list(range(80, 260, 10))


def run_gridsearch(masuite_id):
    run.args.log_params = True
    run.args.overwrite = True
    for epoch in epochs:
        for batch_size in batch_sizes:
            for lr in lrs:
                print(f'Running {epoch} epochs with batch size of {batch_size} and lr of {lr}.')
                run.args.lr = lr
                run.args.batch_size = batch_size
                run.args.num_epochs = epoch
                run.run(masuite_id)


if __name__ == '__main__':
    run_gridsearch('cartpole_simplepg/0')