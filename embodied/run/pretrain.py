import pickle
import time

import elements
import embodied


def pretrain(make_model, make_stream, make_logger, args):

  model = make_model()
  dataset_train = iter(model.stream(make_stream(None, 'train')))
  dataset_report = iter(model.stream(make_stream(None, 'report')))
  dataset_eval = iter(model.stream(make_stream(None, 'eval')))
  logger = make_logger()
  step = logger.step

  should_log = embodied.GlobalClock(args.log_every)
  should_report = embodied.GlobalClock(args.report_every)
  should_save = embodied.GlobalClock(args.save_every)

  train_agg = elements.Agg()
  usage = elements.Usage(**args.usage)
  fps = elements.FPS()

  carry_train = model.init_train(args.batch_size)
  carry_report = model.init_report(args.batch_size)
  carry_eval = model.init_report(args.batch_size)

  cp = elements.Checkpoint(
      elements.Path(args.logdir) / 'checkpoint.pkl',
      write=(args.replica == 0))
  cp.step = step
  cp.model = model
  cp.dataset_train = dataset_train
  cp.dataset_report = dataset_report
  cp.dataset_eval = dataset_eval
  if not cp.exists():
    if args.from_checkpoint:
      data = pickle.loads(elements.Path(args.from_checkpoint).read_bytes())
      model.load(data['model'], regex=args.from_checkpoint_regex)
    cp.save()
  else:
    cp.load()

  print('Starting training')
  while step < args.steps:

    with elements.timer.section('stream'):
      batch = next(dataset_train)
    with elements.timer.section('train'):
      start = time.time()
      carry_train, outs, mets = model.train(carry_train, batch)
      logger.add({'dur/train': time.time() - start})
    train_agg.add(mets)
    step.increment()
    fps.step(args.batch_size * args.batch_length)

    if should_report(step):

      with elements.timer.section('write1'):
        logger.write()

      with elements.timer.section('report'):
        print('Train report')
        start = time.time()
        agg = elements.Agg()
        for _ in range(args.consec_report * args.report_batches):
          carry_report, mets = model.report(carry_report, next(dataset_report))
          agg.add(mets)
        logger.add({'dur/report': time.time() - start})
        logger.add(agg.result(), prefix='report')

      with elements.timer.section('eval'):
        print('Eval report')
        agg = elements.Agg()
        for _ in range(args.consec_report * args.report_batches):
          carry_eval, mets = model.report(carry_eval, next(dataset_eval))
          agg.add(mets)
        logger.add(agg.result(), prefix='eval')

      logger.add({'timer': elements.timer.stats()['summary']})

      with elements.timer.section('write2'):
        logger.write()

    if should_log(step):
      logger.add(train_agg.result(), prefix='train')
      logger.add(usage.stats(), prefix='usage')
      fps_result = fps.result()
      logger.add({'fps': fps_result, 'spf': 1 / fps_result})

    if should_save(step):
      with elements.timer.section('saving'):
        cp.save()

  logger.close()
