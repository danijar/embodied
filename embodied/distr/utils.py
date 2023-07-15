import sys
import time
import traceback
import multiprocessing

import embodied

mp = multiprocessing.get_context('spawn')
PRINT_LOCK = mp.Lock()


def run(workers):
  for worker in workers:
    if not worker.started:
      worker.start()

  while True:

    # print([x.exitcode for x in workers])  # TODO

    if all(x.exitcode == 0 for x in workers):
      print('All workers terminated successfully.')
      return

    for worker in workers:
      if worker.exitcode not in (None, 0):
        print('Terminating worker due to error.')

        # Wait for everybody who wants to print their error messages.
        time.sleep(1)

        for other in workers:
          if other is worker:
            continue
          if hasattr(worker, 'stop'):
            worker.stop(0.1)
            if not worker.alive:
              continue
          worker.terminate()

        msg = f'Terminated workers due to crash in {worker.name}.'
        raise RuntimeError(msg)
    time.sleep(0.1)


def warn_remote_error(e, name, lock=PRINT_LOCK):
  summary = list(traceback.format_exception_only(e))[0].strip('\n')
  full = ''.join(traceback.format_exception(e)).strip('\n')
  msg = f"Exception in worker '{name}' ({summary}). "
  msg += 'Call check() to reraise in main process. '
  msg += f'Worker stack trace:\n{full}'
  with lock:
    embodied.print(msg, 'red')
  if sys.version_info.minor >= 11:
    e.add_note(f'\nWorker stack trace:\n\n{full}')
