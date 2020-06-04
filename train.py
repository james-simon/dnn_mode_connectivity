import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils

import collections

print("qqqqqqq")

def triple(x):
  return(3*x)

TrainArgSet = collections.namedtuple('TrainArgSet', ['dir', 'dataset', 'use_test', 'transform', 'data_path', 'batch_size',
                                     'num_workers', 'model', 'curve', 'num_bends', 'init_start',
                                     'fix_start', 'init_end', 'fix_end', 'init_linear', 'resume', 'epochs',
                                     'save_freq', 'lr', 'momentum', 'wd', 'seed'])
def train_model(dir='/tmp/curve/', dataset='CIFAR10', use_test=True, transform='VGG',
                data_path=None, batch_size=128, num_workers=4, model_type=None, curve_type=None,
                num_bends=3, init_start=None, fix_start=True, init_end=None, fix_end=True,
                init_linear=True, resume=None, epochs=200, save_freq=50, lr=.01, momentum=.9, wd=1e-4, seed=1):
    args = TrainArgSet(dir=dir, dataset=dataset, use_test=use_test, transform=transform,
                data_path=data_path, batch_size=batch_size, num_workers=num_workers, model=model_type, curve=curve_type,
                num_bends=num_bends, init_start=init_start, fix_start=fix_start, init_end=init_end, fix_end=fix_end,
                init_linear=init_linear, resume=resume, epochs=epochs, save_freq=save_freq, lr=lr, momentum=momentum, wd=wd, seed=seed)

    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test
    )

    architecture = getattr(models, args.model)

    if args.curve is None:
        model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    else:
        curve = getattr(curves, args.curve)
        model = curves.CurveNet(
            num_classes,
            curve,
            architecture.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=architecture.kwargs,
        )
        base_model = None
        if args.resume is None:
            for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
                if path is not None:
                    if base_model is None:
                        base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
                    checkpoint = torch.load(path)
                    print('Loading %s as point #%d' % (path, k))
                    base_model.load_state_dict(checkpoint['model_state'])
                    model.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                model.init_linear()
    model.cuda()

    def learning_rate_schedule(base_lr, epoch, total_epochs):
        alpha = epoch / total_epochs
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = factor = .01*(1 - ((alpha - .9)/.1))
        return factor * base_lr


    criterion = F.cross_entropy
    regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd if args.curve is None else 0.0
    )


    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

    utils.save_checkpoint(
        args.dir,
        start_epoch - 1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

    has_bn = utils.check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}
    for epoch in range(start_epoch, args.epochs + 1):

        # if epoch%10 == 0:
        #   print("<***** STARTING EPOCH " + str(epoch) + " *****>")

        time_ep = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        utils.adjust_learning_rate(optimizer, lr)

        train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
        if args.curve is None or not has_bn:
            test_res = utils.test(loaders['test'], model, criterion, regularizer)

        if epoch % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        time_ep = time.time() - time_ep
        values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
                  test_res['accuracy'], time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 1 or epoch == start_epoch:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )
