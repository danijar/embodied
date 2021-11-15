# DreamerV2

**Framework:** TensorFlow 2

```
@article{hafner2020dreamerv2,
  title={Mastering Atari with Discrete World Models},
  author={Hafner, Danijar and Lillicrap, Timothy and Norouzi, Mohammad and Ba, Jimmy},
  journal={arXiv preprint arXiv:2010.02193},
  year={2020}
}
```

- [Official code](https://github.com/danijar/dreamerv2)
- [Blog post](https://ai.googleblog.com/2021/02/mastering-atari-with-discrete-world.html)
- [Research paper](https://arxiv.org/pdf/2010.02193.pdf)

## Features

- **Inputs:** Images, Proprioceptive, Both
- **Actions:** Categorical, Continuous
- **Exploration:** Random, Greedy, Plan2Explore, Model Loss

## Examples

```sh
python3 train.py \
  --logdir ~/logdir/$(date +%Y%m%d-%H%M%S) \
  --configs dmc_vision --task dmc_walker_walk
```

```sh
python3 train.py \
  --logdir ~/logdir/$(date +%Y%m%d-%H%M%S) \
  --configs atari --task atari_pong
```

```sh
python3 train.py \
  --logdir ~/logdir/$(date +%Y%m%d-%H%M%S) \
  --configs crafter
```

```sh
python3 train.py \
  --logdir ~/logdir/$(date +%Y%m%d-%H%M%S) \
  --run train_eval --expl_behavior Plan2Explore
  --configs dmc_vision
```

## Tips

- **Standalone usage.** When you run this agent within the Embodied repository,
  it will load Embodied from the repo. You can also move this agent directory
  anywhere else and use it with `pip3 install embodied`.

- **Efficient debugging.** Add the `debug` config as in `--configs
  atari debug` to reduce the batch size and model size, increase logging, and
  disable `tf.function` JIT compilation for easy line-by-line debugging.

- **Infinite gradient norms.** This is normal and described under loss scaling
  in the [mixed precision][mixed] guide. You can disable mixed precision by
  passing `--precision 32`.

- **Accessing logs.** The metrics are stored both as TensorBoard events and in
  JSON lines format in `logdir/metrics.jsonl`. You can directly load them using
  `pandas.read_json()`.

[mixed]: https://www.tensorflow.org/guide/mixed_precision
