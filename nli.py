from __future__ import print_function, division

import os

# Options before dynet is important for internal reason (being able to specify gpu and stuff)
import options

import dynet as dy
import numpy as np

import data
import vocabulary
import trainers
import util
from classifier import get_classifier
from genre import get_genre_classifier
import models

def train(opt):
    """Train over a training/dev set"""
    timer, log = util.Timer(), util.Logger(opt.verbose)

    # Load data
    train_sents, train_a, train_y, train_g = data.load_snli(opt.train_file, multi_labels=opt.multi_labels, max_length=opt.max_len)
    dev_sents, dev_a, dev_y, _ = data.load_snli(opt.dev_file, max_length=-1)

    # Initialize vocabulqry
    vocab = vocabulary.Vocabulary()
    all_sentences = list(map(lambda x: x[0], train_sents)) + list(map(lambda x: x[1], train_sents))
    if os.path.isfile(opt.vocab_file):
        vocab = vocabulary.load_vocab(opt.vocab_file)
    else:
        vocab.init(all_sentences, train_y ,opt)
        vocabulary.save_vocab(opt.vocab_file, vocab)

    # Data size and classes
    N_train, N_dev = len(train_sents), len(dev_sents)
    N_classes = len(vocab.classes)

    # Map words to IDs (in numpy arrays for advanced indexing)
    train_x = [(vocab.sent_to_ids(w1), vocab.sent_to_ids(w2)) for w1, w2 in train_sents]
    dev_x = [(vocab.sent_to_ids(w1), vocab.sent_to_ids(w2)) for w1, w2 in dev_sents]

    # Map classes to IDs
    train_y = vocab.translate_literal_labels(train_y)
    dev_y = vocab.translate_literal_labels(dev_y)

    # Genres
    genres = {g: i for i, g in enumerate(set(train_g))}
    train_g = np.asarray([genres[g] for g in train_g])
    N_genres = len(genres)

    # Create model
    Model = getattr(models, opt.model_type)
    model = Model(vocab, opt)
    classif = get_classifier(N_classes, model.pc, opt)
    genre_classif = get_genre_classifier(N_genres, model.pc, opt) if opt.adv_weight else None
    # Trainer
    trainer = trainers.get_trainer(opt, model.pc)
    trainer.set_clip_threshold(opt.gradient_clip)

    # Early stopping variables
    best_acc = 0
    deadline = 0

    multi_reg = 0

    # Test biggest batch for memory
    longest_sample = max(train_x, key=lambda x: max(len(x[0]), len(x[1])))
    biggest_batch = [longest_sample for i in range(opt.batch_size)]
    labels = [0 for i in range(opt.batch_size)]
    # Initialize Computation graph
    dy.renew_cg()
    classif.init(test=False, update=True)
    timer.restart()
    # Run model
    h_self, h_cross, hx_loss, hy_loss = model.compare_batch(biggest_batch, None, test=False)
    self_loss = classif.loss_batch(h_self, labels)
    cross_loss = classif.loss_batch(h_cross, labels)
    # penalize the self and cross systems with some ratio
    final_loss = self_loss * opt.self_loss_pct + cross_loss * (1 - opt.self_loss_pct)

    # Backward pass
    final_loss.backward()
    hx_loss.backward()
    hy_loss.backward()
    # If everything went smoothly: print OK
    l_p, l_h = map(len, longest_sample)
    log.info('Memory test passed for sentences of lengths (%d, %d) x %d in %.1f s' % (l_p, l_h, opt.batch_size, timer.tick()))

    # Start training
    for ITER in range(1, opt.num_iters + 1):
        # Shuffle training set
        order = data.shuffle_multinli(len(train_x), percent_snli=15)
        N_train = len(order)
        # Initialize monitoring variables
        loss_value, x_loss, y_loss, n_processed = 0, 0, 0, 0
        timer.restart()
        # Run over the training set
        for b in range(0, N_train, opt.batch_size):
            # Actual batch size (in case we're at the end of the training corpus)
            bsize = min(opt.batch_size, N_train - b)
            # Track number of samples processed
            n_processed += bsize
            # Initialize Computation graph
            dy.renew_cg()
            classif.init(test=False, update=True)
            # Retrieve samples
            words = [train_x[order[i]] for i in range(b, b+bsize)]
            actions = [train_a[order[i]] for i in range(b, b+bsize)]
            labels = train_y[order[b:b+bsize]]
            batch_genres = train_g[order[b:b+bsize]]
            # Run on batch
            h_self, h_cross, hx_loss, hy_loss = model.compare_batch(words, actions, test=False)
            self_loss = classif.loss_batch(h_self, labels)
            cross_loss = classif.loss_batch(h_cross, labels)
            # penalize the self and cross systems with some ratio
            final_loss = self_loss * opt.self_loss_pct + cross_loss * (1 - opt.self_loss_pct)
            # penalize as according to how close they are to each other-- maybe introduce some scaling up system?
            hx_loss *= multi_reg
            hy_loss *= multi_reg
            # I won't be carrying over support for these features so if they break it's rip
            # Add optional l2 reg (relatively slow right now)
            if opt.l2_reg:
                final_loss += opt.l2_reg * dy.esum([dy.squared_norm(p.expr()) for p in model.pc.parameters_list()])
            # Optional adversarial loss
            if opt.adv_weight:
                # Train discriminator
                genre_classif.init(test=False, update=True)
                final_loss += genre_classif.disc_loss_batch(classif.h, batch_genres)
                if opt.adv_start_epoch <= ITER:
                    # After a bunch of epochs start backproping the adversarial loss
                    final_loss += opt.adv_weight * genre_classif.gen_loss_batch(classif.h)
                # Forward pass
            loss_value += final_loss.value() * bsize
            x_loss += hx_loss.value() * bsize
            y_loss += hy_loss.value() * bsize

            # Backward pass
            (final_loss + hx_loss + hy_loss).backward()
            # Update parameters
            trainer.update()
            # Print stats every 10% of the training set
            if (b // opt.batch_size + 1) % ((N_train // opt.batch_size) // 10) == 0:
                # Compute stats
                avg_loss = loss_value / n_processed
                avg_x_loss = x_loss / n_processed
                avg_y_loss = y_loss / n_processed
                elapsed = timer.tick()
                sample_per_sec = n_processed / elapsed
                info = (b // opt.batch_size + 1, avg_loss, avg_x_loss, avg_y_loss, elapsed, sample_per_sec)
                # Print everything
                trainer.status()
                log.info('Loss @%d updates: %.3f (%.3f, %.3f) (%.2fs, %d sample/s)' % info)
                # Re-initialize monitoring variables
                loss_value, x_loss, y_loss, n_processed = 0, 0, 0, 0

        # Evaluate on dev set
        dev_loss = 0
        dev_accuracy = 0
        timer.restart()
        for i in range(N_dev):
            (x1, x2), (a1, a2) = dev_x[i], dev_a[i]
            label = dev_y[i]
            dy.renew_cg()
            classif.init(test=True)
            h, _, _, _ = model.compare(x1, x2, a1, a2, test=True)
            loss = classif.loss(h, label)
            dev_loss += loss.value()
            prediction = np.argmax(classif.score(h).npvalue())
            dev_accuracy += 1 if prediction == label else 0
        avg_loss = dev_loss / N_dev
        avg_acc = dev_accuracy / N_dev * 100
        sample_per_sec = N_dev / timer.tick()
        log.info('Dev accuracy @%d iterations: %.2f%% nll: %.3f (%d sample/s)' % (ITER, avg_acc,  avg_loss,sample_per_sec))
        # Early stopping
        if avg_acc > best_acc:
            best_acc = avg_acc
            # Save model
            model.pc.save(opt.model_file)
            # Reset deadline to 0
            deadline = 0
        else:
            deadline+=1
            if deadline>opt.patience:
                # Stop if patience is exceeded
                log.info('Early stopping at %d iterations, best accuracy: %.2f' % (ITER, best_acc))
                exit()
            else:
                # Otherwise decrease learning rate
                trainer.restart()
                trainer.learning_rate *= 1 - opt.learning_rate_decay
                # Restart from previous best
                model.pc.populate(opt.model_file)

        if ITER >= 3:
            multi_reg += opt.multi_reg
        if ITER % 2 == 0:
            trainer.learning_rate *= 1 - opt.learning_rate_decay

def test(opt):
    """Test model"""
    timer, log = util.Timer(), util.Logger(opt.verbose)

    # Load data
    test_sents, test_a, test_y, _ = data.load_snli(opt.test_file, max_length=-1)

    # Load vocabulqry
    vocab = vocabulary.load_vocab(opt.vocab_file)

    # Map words to IDs
    test_x = [(vocab.sent_to_ids(w1), vocab.sent_to_ids(w2)) for w1, w2 in test_sents]

    # Map classes to IDs
    test_y = map(lambda x: vocab.classes[x], test_y)

    # Classes and number of samples
    N_test = len(test_sents)
    N_classes = len(vocab.classes)

    # Create model
    Model = getattr(models, opt.model_type)
    model = Model(vocab, opt)
    classif = get_classifier(N_classes, model.pc, opt)
    genre_classif = get_genre_classifier(6, model.pc, opt) if opt.adv_weight else None

    # Load model
    model.pc.populate(opt.model_file)

    # Initialize metrics
    test_loss = 0
    test_accuracy = 0
    timer.restart()

    # Start testing
    for i in range(N_test):
        (x1, x2), (a1, a2) = test_x[i], test_a[i]
        dy.renew_cg()
        classif.init(test=True)
        h = model.compare(x1, x2, a1, a2, test=True)
        prediction = classif.score(h).npvalue().argmax()

        label = test_y[i]
        test_accuracy += 1 if prediction== label else 0
    # Print metrics
    avg_loss = test_loss / N_test
    avg_acc = test_accuracy / N_test * 100
    sample_per_sec = N_test / timer.tick()
    log.info('Test accuracy: %.2f%% nll: %.3f (%d sample/s)' % (avg_acc, avg_loss, sample_per_sec))

def predict(opt):
    """Predict classes"""
    timer, log = util.Timer(), util.Logger(opt.verbose)

    # Load data
    test_sents, test_a, pairids = data.load_mnli_unlabeled(opt.test_file)

    # Load vocabulqry
    vocab = vocabulary.load_vocab(opt.vocab_file)
    
    # Classes names
    class_names =  {i:l for l, i in vocab.classes.items()}

    # Map words to IDs
    test_x = [(vocab.sent_to_ids(w1), vocab.sent_to_ids(w2)) for w1, w2 in test_sents]

    # Classes and number of samples
    N_test = len(test_sents)
    N_classes = len(vocab.classes)

    # Create model
    Model = getattr(models, opt.model_type)
    model = Model(vocab, opt)
    classif = get_classifier(N_classes, model.pc, opt)
    genre_classif = get_genre_classifier(6, model.pc, opt) if opt.adv_weight else None

    # Load model
    model.pc.populate(opt.model_file)

    # Initialize predictions
    predictions = ['pairID,gold_label']
    timer.restart()

    # Start testing
    for i in range(N_test):
        (x1, x2), (a1, a2) = test_x[i], test_a[i]
        dy.renew_cg()
        classif.init(test=True)
        h = model.compare(x1, x2, a1, a2, test=True)
        prediction = classif.score(h).npvalue().argmax()
        # Add prediction to output
        predictions.append('%s,%s' % (pairids[i], class_names[prediction]))

    # Elapsed time
    elapsed = timer.tick()
    log.info('Time: %1fs (%d samples/s)' % (elapsed, N_test/elapsed))

    # Save predictions
    util.savetxt(opt.predict_file, predictions)

def sentence_reps(opt):
    """Generate sentence representations"""
    timer, log = util.Timer(), util.Logger(opt.verbose)

    # Load data
    test_sents, test_a, pairids, _ = data.load_snli(opt.test_file, max_length=-1)

    # Load vocabulqry
    vocab = vocabulary.load_vocab(opt.vocab_file)
    
    # Classes names
    class_names =  {i:l for l, i in vocab.classes.items()}

    # Map words to IDs
    test_x = [(vocab.sent_to_ids(w1), vocab.sent_to_ids(w2)) for w1, w2 in test_sents]

    # Classes and number of samples
    N_test = len(test_sents)
    N_classes = len(vocab.classes)

    # Create model
    Model = getattr(models, opt.model_type)
    model = Model(vocab, opt)
    classif = get_classifier(N_classes, model.pc, opt)

    # Load model
    model.pc.populate(opt.model_file)

    # Initialize predictions
    timer.restart()
    hs = []

    # Start testing
    for i in range(N_test):
        (x1, x2), (a1, a2) = test_x[i], test_a[i]
        dy.renew_cg()
        classif.init(test=True)
        h = model.compare(x1, x2, a1, a2, test=True)
        score = classif.score(h)
        hs.append(classif.h.npvalue())

    h_array = np.array(hs)

    # Elapsed time
    elapsed = timer.tick()
    log.info('Time: %1fs (%d samples/s)' % (elapsed, N_test/elapsed))

    # Save
    np.save(opt.sentence_reps_file.format('h'), h_array)

def saliency(opt):
    """Generate sentence representations"""
    timer, log = util.Timer(), util.Logger(opt.verbose)

    # Load data
    test_sents, test_a, test_y, _ = data.load_snli(opt.test_file, max_length=-1)

    # Load vocabulqry
    vocab = vocabulary.load_vocab(opt.vocab_file)
    
    # Classes names
    class_names =  {i:l for l, i in vocab.classes.items()}

    # Map words to IDs
    test_x = [(vocab.sent_to_ids(w1), vocab.sent_to_ids(w2)) for w1, w2 in test_sents]

    # Map classes to IDs
    test_y = map(lambda x: vocab.classes[x], test_y)

    # Classes and number of samples
    N_test = len(test_sents)
    N_classes = len(vocab.classes)

    # Create model
    Model = getattr(models, opt.model_type)
    model = Model(vocab, opt)
    classif = get_classifier(N_classes, model.pc, opt)

    # Load model
    model.pc.populate(opt.model_file)

    # Initialize predictions
    timer.restart()
    hs = []

    extremal_percent=0
    # Start testing
    for i in range(N_test):
        (x1, x2), (a1, a2) = test_x[i], test_a[i]
        dy.renew_cg()
        classif.init(test=True)
        _ = model.compare(x1, x2, a1, a2, test=True)
        if opt.model_type == 'Stacked_BILSTM':
            amax = np.argmax(model.h.npvalue(), axis=1)
            n = amax.size // 2
            extremal_percent += (amax[:, 0]==0).sum()/n + (amax[:, 0]==len(x1)-1).sum()/n 
            extremal_percent += (amax[:, 1]==0).sum()/n + (amax[:, 1]==len(x2)-1).sum()/n 
            if True:
                log.info(' '.join(vocab.ids_to_sent(x1)))
                heat = ['%.2f' % ((amax[:, 0]==i).sum()/n) for i in range(len(x1))]
                heat_fwd = ['%.2f' % (2*(amax[:(n//2), 0]==i).sum()/n) for i in range(len(x1))]
                heat_bwd = ['%.2f' % (2*(amax[(n//2):, 0]==i).sum()/n) for i in range(len(x1))]
                log.info(' '.join(heat))
                log.info(' '.join(heat_fwd))
                log.info(' '.join(heat_bwd))
                #print(','.join(map(str, amax[:, 0].tolist())))

                log.info(' '.join(vocab.ids_to_sent(x2)))
                heat = ['%.2f' % ((amax[:, 1]==i).sum()/n) for i in range(len(x2))]
                heat_fwd = ['%.2f' % (2*(amax[:(n//2), 1]==i).sum()/n) for i in range(len(x2))]
                heat_bwd = ['%.2f' % (2*(amax[(n//2):, 1]==i).sum()/n) for i in range(len(x2))]
                log.info(' '.join(heat_fwd))
                log.info(' '.join(heat_bwd))
                log.info(' '.join(heat))
        elif opt.model_type == 'AttBILSTM':
            att= model.atts.npvalue()
            log.info(' '.join(vocab.ids_to_sent(x1)))
            heat = att[:, :len(x1), 0].mean(axis=0)
            log.info(' '.join(['%.2f' %h for h in heat]))

            log.info(' '.join(vocab.ids_to_sent(x2)))
            heat = att[:, :len(x2), 0].mean(axis=0)
            log.info(' '.join(['%.2f' %h for h in heat]))
        #log.info(','.join(map(str, amax[:, 1].tolist())))
        #loss = classif.loss(h)
        #loss.backward(full=True)
        #grads_1 = [v.grad_as_array()[:, 0] for v in model.input_words]
        #for w1, g1 in zip(x1, grads_1):
        #    print(w1, np.linalg.norm(g1))
        #print('-' * 20)
        #grads_2 = [v.grad_as_array()[:, 1] for v in model.input_words]
        #for w2, g2 in zip(x2, grads_2):
        #    print(w2, np.linalg.norm(g2))
        #print('=' * 20)

    log.info('%.2f' % (100 * extremal_percent / (2*N_test)))
    # Elapsed time
    elapsed = timer.tick()
    log.info('Time: %1fs (%d samples/s)' % (elapsed, N_test/elapsed))

    # Save
    #np.save(opt.sentence_reps_file.format('h'), h_array)

if __name__ == '__main__':
    # Get options
    opt = options.get_options()
    # Perform one of the tasks
    if opt.train:
        train(opt)
    elif opt.test:
        test(opt)
    elif opt.predict:
        predict(opt)
    elif opt.sentence_reps:
        sentence_reps(opt)
    elif opt.saliency:
        saliency(opt)
    else:
        raise ValueError('No known task specified')




