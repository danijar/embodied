import embodied


def agent_server(make_agent, args):
  agent = make_agent()
  cp = embodied.Checkpoint(args.logdir / 'agent.ckpt')
  cp.agent = agent
  cp.load_or_save()
  server = embodied.distr.Server2(args.agent_addr, ipv6=args.ipv6)
  server.bind('policy', agent.policy)
  server.bind('train', agent.train)
  server.bind('report', agent.report)
  server.bind('checkpoint', cp.save)
  server.run()


def replay_server(make_replay, args):
  replay = make_replay()
  cp = embodied.Checkpoint(args.logdir / 'replay.ckpt')
  cp.replay = replay
  cp.load_or_save()
  server = embodied.distr.Server2(args.replay_addr, ipv6=args.ipv6)
  # server.bind('add', lambda data: replay.add(data, data.pop('worker')))
  # server.bind('sample', replay._sample)
  server.bind('add_batch', ...)
  server.bind('sample_batch', ...)
  server.bind('checkpoint', cp.save)
  server.run()


def metrics_server(args):
  logger = ...
  agg = ...
  epstats = ...
  server = embodied.distr.Server2(args.metrics_addr, ipv6=args.ipv6)
  # server.bind('add', lambda data: replay.add(data, data.pop('worker')))
  # server.bind('sample', replay._sample)
  server.bind('add', ...)
  server.bind('log', ...)
  server.run()


def actor_proxy(args):
  agent = embodied.distr.Client(args.agent_addr)
  replay = embodied.distr.Client(args.replay_addr)
  metrics = embodied.distr.Client(args.metrics_addr)

  def act(obs):
    act = agent.policy(obs).result()
    return act, {**obs, **act}

  def store(tran):
    replay.add(tran)
    metrics.add(todo)

  server = embodied.distr.Server2(args.actor_addr)
  server.bind(
      'act', act, store,
      batch=args.actor_batch, workers=args.actor_workers)
  server.run()


def env_client(make_env, i, args):
  env = make_env()
  agent = embodied.distr.Client(args.actor_addr)
  agent.connect()
  act = {'reset': True}
  while True:
    obs = env.step(act)
    act = agent.policy(obs).result()


def learner_client(args):
  agent = embodied.distr.Client(args.agent_addr)
  replay = embodied.distr.Client(args.replay_addr)
  metrics = embodied.distr.Client(args.metrics_addr)
  dataset = batcher(lambda: replay.sample().result(), prefetch=4)
  while True:
    data = next(dataset)
    outs, metrics = agent.train(data)
