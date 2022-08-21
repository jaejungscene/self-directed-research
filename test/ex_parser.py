import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='pytorch-cifar-examples', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wandb', type=str, default='true', 
                    help='choose activating wandb')
    # parser.add_argument('data_dir', type=str, help='dataset dir')
    # parser.add_argument('data_dir02', type=str, help='dataset dir')
    # parser.add_argument('--data_type', type=str, dest='dataType', help='data type')

    # train_input = parser.add_argument_group('train_input')
    # parser.add_argument('--name', type=str, help='name')
    # parser.add_argument('--age', type=int, help='age')

    # test_input = parser.add_argument_group('train_input')
    # parser.add_argument('--univ', type=str, help='univ')
    # parser.add_argument('--sex', type=str, help='sex')

    parser.set_defaults(dataType='json')

    return parser

if __name__ == '__main__':
    args_parser = get_args_parser()
    # print(args_parser.print_help())
    args = args_parser.parse_args()
    print()
    # print(args)
    # print(args.data_dir)
    # print(args.name)
