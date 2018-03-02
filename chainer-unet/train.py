import argparse

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
from updater import FacadeUpdater
from net import Encoder
from net import Decoder
from img_dct_dataset import ImgDCTDataset

def main():
    parser = argparse.ArgumentParser(description="chainer implementation of Unet")
    parser.add_argument("--batchsize", "-b", type=int, default=1,
                        help="Number of images in each mini-batch")
    parser.add_argument("--epoch", "-e", type=int, default=200,
                        help="epoch")
    parser.add_argument("--gpu", "-g", type=int, default=-1,
                        help="GPU ID")
    parser.add_argument("--dataset", "-i", default="./train/",
                        help="Directory of image files")
    parser.add_argument("--out", "-o", default="result/",
                        help="Directory to output the result")
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Only Encoder Decoder with Unet
    enc = Encoder(in_ch=3) # in_ch => 3(YCbCr)
    dec = Decoder(out_ch=3) # out_ch => 3(DCT)

    # GPU set up
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        enc.to_gpu()
        dec.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)

    train_d = ImgDCTDataset(args.dataset, data_range=(0, 1000))
    test_d = ImgDCTDataset(args.dataset, data_range=(1000, 2000))
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize)

    updater = FacadeUpdater(
        models=(enc, dec),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec },
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
                   trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(
    #     dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'dec/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == "__main__":
    main()