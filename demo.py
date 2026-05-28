from main import parse, train, report
from src.model import SMD_DDA

if __name__=="__main__":
    name = "Gdataset"
    nums = 7


    args = parse(print_help=True)
    args.dataset_name = name
    args.epochs = 1000
    if name=="lagcn":
        args.edge_dropout=0.5
        # args.embedding_dim=64
    else:
        args.edge_dropout=0.2
    args.drug_neighbor_num = 7
    args.disease_neighbor_num = 7
    args.lr = 0.006
    args.n_splits = 10
    #args.loss_fn = 'focal'
    train(args, SMD_DDA)
    # report("runs")
