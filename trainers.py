from __future__ import print_function, division
import sys
import dynet as dy


def get_trainer(opt, pc):
    """Returns trainer from string"""
    if opt.trainer == 'sgd':
        trainer = dy.SimpleSGDTrainer(pc, learning_rate=opt.learning_rate)
    elif opt.trainer == 'clr':
        trainer = dy.CyclicalSGDTrainer(pc, learning_rate_min=opt.learning_rate / 10.0,
                                        learning_rate_max=opt.learning_rate)
    elif opt.trainer == 'momentum':
        trainer = dy.MomentumSGDTrainer(pc, learning_rate=opt.learning_rate)
    elif opt.trainer == 'adagrad':
        trainer = dy.AdagradTrainer(pc, learning_rate=opt.learning_rate)
    elif opt.trainer == 'rmsprop':
        trainer = dy.RMSPropTrainer(pc, learning_rate=opt.learning_rate)
    elif opt.trainer == 'adam':
        trainer = dy.AdamTrainer(pc, opt.learning_rate)
    else:
        print('Trainer name invalid or not provided, using SGD', file=sys.stderr)
        trainer = dy.SimpleSGDTrainer(pc, learning_rate=opt.learning_rate)

    trainer.set_clip_threshold(opt.gradient_clip)

    return trainer