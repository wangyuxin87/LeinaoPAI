# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on one parameter server (ps), while the ops
are executed on two worker nodes by default. The TF sessions also run on the
worker node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_string("train_dir", "/tmp/log-data",
                    "Directory for storing training data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 1,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts","localhost:1001",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:1002,localhost:1003",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")

FLAGS = flags.FLAGS


IMAGE_PIXELS = 28

def getRoleSpec(env_dict, role_num, role_spec, role):
  for i in range(role_num):
    env_ps_port_list = "PAI_PORT_LIST_{}_{}".format(role, i)
    port_list = env_dict[env_ps_port_list].split(',')

    print ('--------------------')
    print ('PAI_CONTAINER_HOST_IP = %s' % (env_dict['PAI_CONTAINER_HOST_IP']))
    print ('port_list[0] = %s' % (port_list[0]))
    print ('PAI_CURRENT_TASK_ROLE_NAME = %s' % (env_dict['PAI_CURRENT_TASK_ROLE_NAME']))
    print ('role = %s' % (role))
    print ('PAI_TASK_INDEX = %s' % (env_dict['PAI_TASK_INDEX']))
    print ('str(i) = %s' % (str(i)))
    print ('--------------------')
    print (' ')
    
    if env_dict['PAI_CONTAINER_HOST_IP'] == port_list[0] and env_dict['PAI_CURRENT_TASK_ROLE_NAME'] == role and env_dict['PAI_TASK_INDEX'] == str(i):
      ip_index = role_spec.index(port_list[0])
      role_spec[ip_index] = role_spec[ip_index] + ":10001"
      continue
    ip_index = role_spec.index(port_list[0])
    role_spec[ip_index] = role_spec[ip_index] + ":" + port_list[1]
  return role_spec


def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  if FLAGS.download_only:
    sys.exit(0)

  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name, end='  ')
  print("task index = %d" % FLAGS.task_index)
  env_dict = os.environ
  
  #Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")
  print('ps_spec = ', end='')
  print(ps_spec)
  print('worker_spec = ', end='')
  print(worker_spec)  
  
  ps_num = len(env_dict['PAI_TASK_ROLE_ps_HOST_LIST'].split(","))
  worker_num = len(env_dict['PAI_TASK_ROLE_worker_HOST_LIST'].split(","))
  print("ps_num:{}".format(ps_num)) 
  print("worker_num:{}".format(worker_num))
  ps_spec = getRoleSpec(env_dict, ps_num, ps_spec, "ps")
  worker_spec = getRoleSpec(env_dict, worker_num, worker_spec, "worker")

  # Print the results
  print('ps_spec = ', end='')
  print(ps_spec)
  print('worker_spec = ', end='')
  print(worker_spec)
  
  # Get the number of workers
  num_workers = len(worker_spec)
  print('num_workers = %d' % num_workers)

  # Build the cluster
  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})


  # Create and start a server for the local task.
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, protocol='grpc+verbs')

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Variables of the hidden layer
      hid_w = tf.Variable(
          tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                              stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

      # Variables of the softmax layer
      sm_w = tf.Variable(
          tf.truncated_normal([FLAGS.hidden_units, 10],
                              stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
          name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y_ = tf.placeholder(tf.float32, [None, 10])

      hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      hid = tf.nn.relu(hid_lin)

      y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
      loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 100000 steps have completed.
      step = 0
      start = time.time()
      print("Training starts!")
      
      while not sv.should_stop() and step < 100000:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}

        _, step = sess.run([train_op, global_step], feed_dict=train_feed)

        if step % 50 == 0:
          print("step:", step)
    # Ask for all the services to stop.
    end = time.time()
    print("Total_Time:", end - start)
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
