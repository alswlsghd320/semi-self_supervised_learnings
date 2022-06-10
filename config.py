import argparse


def getConfig():
    parser = argparse.ArgumentParser()
    # Path settings
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--file_name', type=str, default='train_df.csv')
    parser.add_argument('--save_path', type=str, default='results/')

    # Model parameter settings
    parser.add_argument('--encoder_name', type=str, default='resnet50')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--drop_path_rate', type=float, default=0)

    # Training parameter settings
    ## Base Parameter
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--optimizer', type=str, default='Lamb')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    ## Scheduler
    parser.add_argument('--scheduler', type=str, default='cos')
    parser.add_argument('--warm_epoch', type=int, default=5)  # WarmUp Scheduler

    ### Cosine Annealing
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--tmax', type=int, default=145)

    ### MultiStepLR
    parser.add_argument('--milestone', type=int, nargs='*', default=[50])
    parser.add_argument('--lr_factor', type=float, default=0.1)

    ### OnecycleLR
    parser.add_argument('--max_lr', type=float, default=1e-3)

    ## etc.
    parser.add_argument('--patience', type=int, default=10, help='Early Stopping')
    parser.add_argument('--use_tensorboard', type=bool, default=True)
    parser.add_argument('--label_smoothing', type=float, default=0)

    parser.add_argument('--method', type=str, default=None)
    # Semi-supervised Learning
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--lambda_u', type=float, default=1)
    parser.add_argument('--temperature', type=float, default=1)

    ## Noisy Student
    parser.add_argument('--steps', type=int, default=5)

    ## Meta Pseudo Labels


    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = getConfig()
    args = vars(args)
    print(args)