import argparse

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

def main():
    parser = argparse.ArgumentParser(description="chainer implementation of Unet")
    parser.add_argument("--batchsize", "-b", type=int, default=1,
                        help="Number of images in each mini-batch")
    parser.add_argument("--epoch", "-e", type=int, default=200,
                        help="epoch")
    parser.add_argument("--gpu", "-g", type=int, default=-1,
                        help="GPU ID")
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Only Encoder Decoder with Unet
    enc = Encoder(in_ch=3) # in_ch => 3(YCbCr)
    dec = Decoder(out_ch=3) # out_ch => 3(DCT)

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)

    


    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        enc.to_gpu()
        dec.to_gpu()




if __name__ == "__main__":
    main()