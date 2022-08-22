import os
import torch
import shutil

def printSave_end_state(args, best_err1, best_err5, total_time):
    print('='*100)
    print('Best accuracy (top-1 and 5 error):\t', best_err1, 'and',best_err5)
    print('total time:\t', total_time)

    directory = "log/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(directory+'log.txt', 'a') as file:
        file.write('-'*100)
        content = \
            (   
                '\nBest accuracy (top-1 and 5 error):\t{} and {}\n'+\
                'total time:\t{}'
            ).format(
                best_err1, best_err5,
                total_time
            )
        file.write(content)
        file.close()



def printSave_start_condition(args, num_param):
    print(args.expname)
    if args.net_type.endswith('resnet'):
        print("=> creating model '{}'".format(args.net_type+str(args.depth)))
    else:
        print("=> creating model '{}'".format(args.net_type))
    print("=> input size:\t'{}'".format(args.insize))
    print("=> dataset:\t'{}'".format(args.dataset))
    print("=> batch size:\t'{}'".format(args.batch_size))
    print("=> epochs:\t'{}'".format(args.epochs))
    print("=> running rate:\t'{}'".format(args.lr))
    print("=> optimizer:\t'{}'".format(args.optim))
    print('=> the number of model parameters: {:,}'.format(num_param))

    directory = "log/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(directory+'log.txt', 'a') as file:
        content = \
            (   
                'model:\t\t\t{}\n'+\
                'input size:\t\t{}\n'+\
                'dataset:\t\t{}\n'+\
                'batch size:\t\t{}\n'+\
                'epochs:\t\t\t{}\n'+\
                'the number of model parameters:\t{}\n'
            ).format(
                args.net_type+str(args.depth),
                args.insize,
                args.dataset,
                args.batch_size,
                args.epochs,
                num_param
            )
        file.write(content)
        file.close()



def printSave_one_epoch(epoch, args, batch_time, data_time, top1, top5, losses, train=True):
    mode_str = None
    if train:
        mode_str = 'Train'
        print('='*100)
    else:
        mode_str = 'Test'
    
    print(('{}\t'+
          '* Epoch[{}/{}]\t'+
          'Time: ({batch_time.avg:.3f})\t'+
          'Data loading tiem: ({data_time.avg:.3f})\t'+
          'Top 1-err: {top1.avg:.3f}\t'+
          'Top 5-err: {top5.avg:.3f}\t'+
          'Loss: {loss.avg:.3f}').format(
        mode_str, epoch+1, args.epochs, 
        batch_time=batch_time, data_time=data_time,
        top1=top1, top5=top5, loss=losses))
    
    directory = "log/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory+'log.txt', 'a') as file:
        if train:
            file.write('='*100)
        content = \
            ('\n{}\t* Epoch[{}/{}]\t'+\
            'Time: ({batch_time.avg:.3f})\t'+\
            'Data loading tiem: ({data_time.avg:.3f})\t'+\
            'Top 1-err: {top1.avg:.3f}\t'+\
            'Top 5-err: {top5.avg:.3f}\t'+\
            'Loss: {loss.avg:.3f}\n').format(
            mode_str, epoch+1, args.epochs, 
            batch_time=batch_time, data_time=data_time,
            top1=top1, top5=top5, loss=losses)
        file.write(content)
        file.close()    



def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    directory = "log/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'log/%s/' % (args.expname) + 'model_best.pth.tar')