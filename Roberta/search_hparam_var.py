from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import copy
import time
import queue
import json, itertools, collections
import multiprocessing
from multiprocessing import Process, Manager

from run_classifier import args, main
#from run_classifier_dev_b import args, main

_args = args
_sys_stdout = sys.stdout

class EndSignal(object):
  pass

class ChoiceParameter(object):
  def __init__(self, name, values):
    self._name = name
    self._values = values

  @property
  def name(self):
    return self._name

  def __iter__(self):
    return iter(self._values)

  @property
  def setting_count(self):
    return len(self._values)

  def __str__(self):
    return "Parameter '{}': {}".format(self._name, self._values)

class ParameterManager(object):
  def __init__(self):
    self._supported_param_types = {
        "choice": self._choice_param
    }

  def from_dict(self, param_desc):
    if "type" not in param_desc:
      raise ValueError("Invalid parameter description {}, "
                       "'type' missing.".format(param_desc))
    if param_desc["type"] not in self._supported_param_types:
      raise ValueError("Unsupported parameter type {}, "
                       "currently only '{}' supported.".format(
                           param_desc["type"],
                           self._supported_param_types.keys()))
    else:
      return self._supported_param_types[param_desc['type']](param_desc)

  def _choice_param(self, param_desc):
    if 'values' not in param_desc:
      raise ValueError("Invalid ChoiceParameter {}, "
                       "'values' missing.".format(param_desc))
    if not isinstance(param_desc['values'], collections.Iterable):
      raise ValueError("For ChoiceParameter, 'values' should be iterable. ")
    return ChoiceParameter(param_desc['name'], param_desc['values'])

class GridSearchTuner(object):
  def __init__(self):
    self._total_conf = 0
    self._generated_conf = 0
    self._params = {}
    self._param_manager = ParameterManager()

  def from_json(self, json_file):
    with open(json_file) as data_file:
      params = json.loads(data_file.read())
      conf_count = 1
      for param in params:
        if param['name'] in self._params:
          raise ValueError("Duplicated param {} found.".format(param['name']))
        param_obj = self._param_manager.from_dict(param)
        conf_count *= param_obj.setting_count
        self._params[param_obj.name] = param_obj

    self._total_conf = conf_count

    print("-" * 80)
    for key, value in self._params.items():
      print(value)

    print("")
    print("* Total conf count to try: {}".format(self._total_conf))

  @property
  def total_conf(self):
    return self._total_conf

  @property
  def generated_conf(self):
    return self._generated_conf

  @property
  def confs(self):
    param_names = self._params.keys()
    param_objs = self._params.values()

    for values in itertools.product(*param_objs):
      self._generated_conf += 1
      yield dict(zip(param_names, values))

class Logger(object):
  def __init__(self, output_file):
    self.log = open(output_file, "w")

  def write(self, message):
    self.log.write(message)
    self.log.flush()

class Job(object):
  def __init__(self, job_id, gpu_id, settings):
    self._job_id = job_id
    self._gpu_id = gpu_id
    self._settings = settings
    self.is_succeed = False
    self.exception = None
    self.ex_info = None
    self.result = None
    self.log_file = None

  @property
  def settings(self):
    return self._settings

  @property
  def gpu_id(self):
    return self._gpu_id

  @property
  def job_id(self):
    return self._job_id

  def print_user_flags(self, flags, line_limit=80):
    print("-" * 80)

    for flag_name in dir(flags):
      value = "{}".format(getattr(flags, flag_name))
      log_string = flag_name
      log_string += "." * (line_limit - len(flag_name) - len(value))
      log_string += value
      print(log_string)

  def prepare(self):
    cur_process = multiprocessing.current_process()
    print("-" * 80)
    print("* Start to execute job {}".format(self._job_id))
    print("* GPU {} occupied by {}".format(self._gpu_id, cur_process.name))
    if not os.path.exists(_args.log_dir):
        try:
            os.makedirs(_args.log_dir)
        except:
            print("catch a error")
    log_file = os.path.join(_args.log_dir, "job_{}.log".format(self._job_id))
    print("* Log file located at {}".format(log_file))
    print("* Settings: {}".format(self._settings))
    sys.stdout = Logger(log_file)
    self.log_file = log_file

  def execute(self):
    flags = copy.deepcopy(_args)
    cur_process = multiprocessing.current_process()
    flags.output_dir = os.path.join(flags.output_dir)
    #flags.output_dir = os.path.join(flags.output_dir, cur_process.name)
    print("flags.output_dir",flags.output_dir)
    self.update_flags(self._settings, flags)

    os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_id
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    self.print_user_flags(flags)
    self.result = main(flags)

  def update_flags(self, params, FLAGS):
    for k, v in params.items():
      assert hasattr(FLAGS, k), "Invalid parameter {} to update".format(k)
      flags_val_type = type(getattr(FLAGS, k))
      if isinstance(getattr(FLAGS, k), str) and isinstance(v, unicode):
        v = v.encode("ascii")

      assert flags_val_type == type(v), \
              "Inconsistent type FLAGS: {} vs. {}".format(flags_val_type, type(v))
      setattr(FLAGS, k, v)

  '''
  def __cmp__(self, other):
    if not self.is_succeed and not other.is_succeed:
      print("no job succeed")
      return 1
    elif not self.is_succeed: return 1
    elif not other.is_succeed: return -1
    else:
      if self.result > other.result: return -1
      else: return 1
  '''

def evaluate_task(job_q, done_q, gpu_q, lock):
  while True:
    job = job_q.get()
    if isinstance(job, EndSignal):
      break;
    try:
      with lock:
        job.prepare()
      job.execute()
    except Exception as e:
      job.is_succeed = False
      job.exception = str(e)
      job.ex_info = type(e).__name__
    else:
      job.is_succeed = True
    finally:
      gpu_q.put(job.gpu_id)
      done_q.put(job)
      # write to log file
      if not job.is_succeed:
        print("job.log_file",job.log_file)
        with open(job.log_file, "a") as fout:
          fout.write("{}\n".format(job.exception))
      sys.stdout = _sys_stdout
      with lock:
        print("-" * 80)
        if job.is_succeed:
          print("* Job {} succeed, acc: {}".format(job.job_id, job.result))
        else:
          print("! Job {} failed".format(job.job_id))
          print("! Exception: {}".format(job.ex_info))
        print("* Release GPU {}".format(job.gpu_id))

  job_q.put(EndSignal())

if __name__ == '__main__':
  gpu_ids = _args.available_gpus
  gpu_ids = args.available_gpus
  #gpu_ids = '3'
  if _args.model_name_or_path == 'roberta-base':
    num_gpu = len(gpu_ids)
  else:
    num_gpu = int(len(gpu_ids)/2)
  print("num_gpu=",len(gpu_ids),num_gpu)

  tuner = GridSearchTuner()
  tuner.from_json(_args.conf_file)
  job_count = 0

  manager = Manager()
  job_q = manager.Queue(num_gpu)
  done_q = manager.Queue()
  gpu_q = manager.Queue(num_gpu)
  lock = manager.Lock()
  finished_jobs = []

  # fill gpu resources
  if _args.model_name_or_path == 'roberta-base':
    for gpu_id in gpu_ids:
      gpu_q.put(gpu_id)
  else:
    gpu_q.put('0,1')

  workers = [
      Process(target=evaluate_task, args=(job_q, done_q, gpu_q, lock))
      for _ in range(num_gpu)
  ]

  for w in workers:
    w.daemon = True
    w.start()

  conf_iter = tuner.confs
  while True:
    try:
      gpu_id = gpu_q.get_nowait()
    except queue.Empty:
      time.sleep(0.01)
      new_done = False
      while not done_q.empty():
        finished_jobs.append(done_q.get())
        new_done = True
      if new_done:
        print("finished_jobs",finished_jobs)
        # sort
        finished_jobs.sort(key=lambda t: (t.result[1], t.result[2]), reverse=True)

        with lock:
          print("-" * 80)
          print("* Progress: {}/{}".format(len(finished_jobs), tuner.total_conf))
          if finished_jobs[0].is_succeed:
            print("* Best acc: {}, Job: {}".format(finished_jobs[0].result[0],
                                                   finished_jobs[0].job_id))
            print("* Best settings: {}".format(finished_jobs[0].settings))
        # save to result_file
        result_file = os.path.join(_args.output_dir, "tune_results_"+ args.task_name+"_" + args.model_name_or_path +".txt")
        with open(result_file, 'w') as fout:
          for job in finished_jobs:
            if job.is_succeed:
              fout.write("{}\t{}\t{}\t{}\t{}\n".format(job.job_id, job.result[0], job.result[1], job.result[2], job.settings))
            else:
              fout.write("{}\t{}\t{}\n".format(job.job_id, job.ex_info, job.settings))
    else:
      try:
        conf = next(conf_iter)
        job = Job(job_count, gpu_id, conf)
        job_q.put(job)
        job_count += 1
      except StopIteration as e:
        break

  job_q.put(EndSignal())
  for w in workers:
    w.join()

  while not done_q.empty():
    finished_jobs.append(done_q.get())
  # sort
  finished_jobs.sort(key=lambda t: (t.result[1], t.result[2]), reverse=True)

  print("-" * 80)
  print("* All job finished")
  if len(finished_jobs) > 0 and finished_jobs[0].is_succeed:
    print("* Best acc: {}, Job: {}".format(finished_jobs[0].result[0],
                                           finished_jobs[0].job_id))
    print("* Best settings: {}".format(finished_jobs[0].settings))

  # save to result_file
  result_file=os.path.join(_args.output_dir,"tune_results_"+ args.task_name+"_" + args.model_name_or_path +".txt")
  with open(result_file, 'w') as fout:
    for job in finished_jobs:
      if job.is_succeed:
        fout.write("{}\t{}\t{}\t{}\t{}\n".format(job.job_id, job.result[0], job.result[1], job.result[2], job.settings))
      else:
        fout.write("{}\t{}\t{}\n".format(job.job_id, job.ex_info, job.settings))
